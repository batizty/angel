package com.tencent.angel.spark.ml.sparse

import breeze.optimize.DiffFunction
import com.tencent.angel.spark.ml.common.OneHot
import com.tencent.angel.spark.ml.common.OneHot.OneHotVector
import com.tencent.angel.spark.models.vector.BreezePSVector
import org.apache.spark.rdd.RDD

/**
  * Proximal-FTRL for Angel Parameter Server Computing Framework
  *
  */
object ProximalFTRL {

  case class FTRLParams(alpha: Double = 1.0, beta: Double = 1.0, regParamL1: Double = 1.0, regParamL2: Double = 1.0)

  /**
    * Compute Weight through n-vector, z-vector
    *
    * @param param ftrl parameter including(α, β, L1, L2)
    * @param n     n-vector
    * @param z     z-vector
    * @return weight-vector this data set
    */
  def getWeight(param: FTRLParams, n: Array[Double], z: Array[Double]): Array[Double] = {
    z.zip(n) map { case (zi, ni) =>
      val sign = if (zi < 0) -1 else 1
      if ((sign * zi) <= param.regParamL1) {
        0.0
      } else {
        (sign * param.regParamL1 - zi) / ((param.beta + math.sqrt(ni)) / param.alpha + param.regParamL2)
      }
    }
  }


  /**
    * Impl Proximal-FTRL with Parameter Server Framework
    *
    * @param trainData training data set
    */
  case class PSCost(trainData: RDD[(OneHotVector, Double)]) extends DiffFunction[(FTRLParams, BreezePSVector, BreezePSVector)] {

    override def calculate(nzp: (FTRLParams, BreezePSVector, BreezePSVector)): (Double, (FTRLParams, BreezePSVector, BreezePSVector)) = {
      val (ftrlParams: FTRLParams, nvector0: BreezePSVector, zvector0: BreezePSVector) = nzp
      val nvector = nvector0.proxy.getPool().createZero().mkBreeze()
      val zvector = zvector0.proxy.getPool().createZero().mkBreeze()

      val sampleNum = trainData.count()

      val cumLoss: Double = trainData.mapPartitions { iter =>
        val localN: Array[Double] = nvector.toRemote.pull()
        val localZ: Array[Double] = nvector.toRemote.pull()

        val deltaN = new Array[Double](localN.length)
        val deltaZ = new Array[Double](localZ.length)
        val lossSum: Double = iter.map {
          case (features: OneHotVector, label: Double) =>
            val weight: Array[Double] = getWeight(ftrlParams, localN, localZ)
            val margin: Double = -1.0 * OneHot.dot(features, weight)
            val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label
            val loss =
              if (label > 0.0) {
                // log1p is log(1+p) but more accurate for small p
                math.log1p(math.exp(margin))
              } else {
                math.log1p(math.exp(margin)) - margin
              }

            // TODO could be implement by breeze vector computing
            features foreach { index =>
              val gi = gradientMultiplier
              val ni = localN(index)
              val wi = weight(index)
              val sigma = (math.sqrt(ni + gi * gi) - math.sqrt(ni)) / ftrlParams.alpha
              val dn = gi - sigma * wi
              val dz = gi * gi
              localN(index) = localN(index) + dn
              localZ(index) = localZ(index) + dz
              deltaN(index) = deltaN(index) + dn
              deltaZ(index) = deltaZ(index) + dz
            }
            loss
        }.sum

        nvector.toRemote.incrementAndFlush(deltaN)
        zvector.toRemote.incrementAndFlush(deltaZ)

        Iterator.single(lossSum)
      }.sum

      println(s"loss: ${cumLoss / sampleNum}")
      (cumLoss / sampleNum, (ftrlParams, nvector, zvector))
    }
  }

}
