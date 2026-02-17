error id: file://<WORKSPACE>/src/main/scala/project1/WineQualityRegression.scala:
file://<WORKSPACE>/src/main/scala/project1/WineQualityRegression.scala
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:
	 -scalation/theory/data.
	 -data.
	 -scala/Predef.data.
offset: 318
uri: file://<WORKSPACE>/src/main/scala/project1/WineQualityRegression.scala
text:
```scala
package project1

import scalation.theory._

object WineQualityRegression {
  val filePath = "wine_data/wine-quality-red.csv"
  val dataName = "WineQualityRegression"
  val data: Dataset = Dataset(dataName, filePath, 8, Array(0, 1, 2, 3, 4, 5, 6), 7)

  @main def run(): Unit = {
    val utils = new Project1Utils(data@@.x, data.y, data.fname)
//    utils.runLinearRegressions()
//    utils.runRidgeRegressions()
//    utils.runLassoRegressions()
//    utils.runTransformedRegressions()
//    utils.runSymRidgeRegression()
//    utils.runForwardSelect()
//    utils.runBackwardsElimination()
    utils.runStepwiseSelect()
  }
}


```


#### Short summary: 

empty definition using pc, found symbol in pc: 