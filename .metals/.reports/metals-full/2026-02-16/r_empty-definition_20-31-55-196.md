error id: file://<WORKSPACE>/src/main/scala/project1/HousePriceRegression.scala:
file://<WORKSPACE>/src/main/scala/project1/HousePriceRegression.scala
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:
	 -scalation/theory/filePath.
	 -scalation/theory/filePath#
	 -scalation/theory/filePath().
	 -filePath.
	 -filePath#
	 -filePath().
	 -scala/Predef.filePath.
	 -scala/Predef.filePath#
	 -scala/Predef.filePath().
offset: 254
uri: file://<WORKSPACE>/src/main/scala/project1/HousePriceRegression.scala
text:
```scala
package project1

//import scalation.mathstat._
import scalation.theory._

object HousePriceRegression {
  val filePath = "project1/house_price_regression_dataset.csv"
  val dataName = "HousePriceRegression"
  val data: Dataset = Dataset(dataName, filePa@@th, 8, Array(0, 1, 2, 3, 4, 5, 6), 7)

  @main def run(): Unit = {
    val utils = new Project1Utils(data.x, data.y, data.fname)
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