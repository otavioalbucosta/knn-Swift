
import UIKit
import Accelerate
import TabularData
import simd

typealias Weight = (Float, Float, Float, Float, Float)

typealias Iris = (atributo: SIMD4<Float>, rotulo: String)

func findData(dataFrame: DataFrame) -> [Iris] {
        var irisData: [Iris] = dataFrame.rows.map({ row in
            
            return Iris(atributo: SIMD4<Float>(row[1, Float.self]!, row[2, Float.self]!, row[3, Float.self]!, row[4, Float.self]!), rotulo: row[5, String.self]!)
        })
    irisData.shuffle()
        return irisData

}
func readCSV(filename: String) -> DataFrame{
    var dataFrame: DataFrame!
    do{
        let fileURL = Bundle.main.url(forResource: filename, withExtension: "csv")
        let options = CSVReadingOptions(hasHeaderRow: true, delimiter: ",")
        dataFrame = try DataFrame(contentsOfCSVFile: fileURL!, types: ["SepalLengthCm": .float, "SepalWidthCm": .float, "PetalLengthCm": .float, "PetalWidthCm": .float, "Species": .string], options: options)

    }catch {
        print(error)
    }
    return dataFrame
}

func getTrainAndTest(data: DataFrame, proportion: Double) -> ([Iris],[Iris]) {
    var sepalNames = data.summary(of: "Species")[row: 0]["mode", Array<Any>.self]

    var trainData = DataFrame(data.prefix(0))
    var testData = DataFrame(data.prefix(0))
    
    for sepal in sepalNames! {
        let filtered = data.filter(on: "Species", String.self, {$0! == sepal as! String})
        let (train, test) = filtered.randomSplit(by: proportion)

            
            trainData.append(train)
            testData.append(test)
    }

    return (findData(dataFrame: trainData), findData(dataFrame: testData))
}


struct Perceptron {
    var weights: Weight
    var specie: String
    var epochs: Int
    var learningRate: Float
    
    
    init(epochs: Int = 100, learningRate: Float = 0.1, specie: String = "Iris-setosa")
    {
        self.epochs = epochs
        self.learningRate = learningRate
        self.specie = specie
        self.weights = (0,0,0,0,0)
        
        
    }
    
    mutating func train(data: [Iris]) {
        var aux = data
        aux.shuffle()
        for i in 0..<epochs {
            var count = 0
            for iris in aux {
                let isSpecie = iris.rotulo == specie ? 1 : 0
                
                let output = predict(iris: iris)
                let error = Float(isSpecie - output)
                if error == 0 {
                    count += 1
                    continue
                }
                
                weights.0 += learningRate * error * -1
                weights.1 += learningRate * error * iris.atributo.x
                weights.2 += learningRate * error * iris.atributo.y
                weights.3 += learningRate * error * iris.atributo.z
                weights.4 += learningRate * error * iris.atributo.w
                print(weights)
                
            }
            aux.shuffle()
            if count == aux.count {
                print("Break on epoch \(i)")
                break
            }
        }
    }
    

    
    func predict(iris: Iris) -> Int {
        let i1 = -1 * weights.0
        let i2 = iris.atributo.x * weights.1
        let i3 = iris.atributo.y * weights.2
        let i4 = iris.atributo.z * weights.3
        let i5 = iris.atributo.w * weights.4
        
        let u = i1+i2+i3+i4+i5
        return degree(u)
        
    }
    
    
    func degree(_ number: Float) -> Int {
        return number >= 0.0 ? 1 : 0
    }
    
}


var perceptron = Perceptron()
perceptron.epochs = 3000
perceptron.specie = "Iris-virginica"
var data = readCSV(filename: "Iris")
var (trainData, testData) = getTrainAndTest(data: data, proportion: 0.5)
perceptron.train(data: trainData)

var count: Float = 0.0
for singleTest in testData {
    let isSpecie = singleTest.rotulo == perceptron.specie ? 1 : 0
    if perceptron.predict(iris: singleTest) == isSpecie{
        count += 1
    }
}
print("A porcentagem de acertos é: \((count / Float(testData.count)) * 100)%")
