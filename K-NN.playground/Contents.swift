import UIKit
import Accelerate
import TabularData
import simd

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
    print(dataFrame)
    return dataFrame
}

func getTrainAndTest(data: DataFrame, proportion: Double) -> ([Iris],[Iris]) {
    var sepalNames = data.summary(of: "Species")[row: 0]["mode", Array<Any>.self]

    var trainData = DataFrame(data.prefix(0))
    var testData = DataFrame(data.prefix(0))
    
    for sepal in sepalNames! {
        let filtered = data.filter(on: "Species", String.self, {$0! == sepal as! String})
//        print(sepal)
//        print(filtered)
        let (train, test) = filtered.randomSplit(by: proportion)
//        print(train)
//        print(test)
            
            trainData.append(train)
            testData.append(test)
    }

    return (findData(dataFrame: trainData), findData(dataFrame: testData))
}



struct KNN {
    
    var K: Int
    var trainingData: [Iris] = []
    
    mutating func train(data: [Iris]){
        self.trainingData.append(contentsOf: data)
    }
    
    func predict(value: Iris) -> Bool{
        var distances = [Float:String]()
        var nearestNeighbors = [Float:String]()
        
        for data in self.trainingData {
            let dist = distance(data.atributo, value.atributo)
            distances[dist] = data.rotulo
        }
        let sortedKeys = Array(distances.keys).sorted(by: <)
        for i in 0..<self.K {
            nearestNeighbors[sortedKeys[i]] = distances[sortedKeys[i]]
        }
        
        var counts = [String: Int]()
        nearestNeighbors.values.forEach({counts[$0] = (counts[$0] ?? 0) + 1})
        
        if let (label, _) = counts.max(by: {$0.1 > $1.1}) {
            print("\(value) is \(label)")
            print(label == value.rotulo)
            return label == value.rotulo
        }
        print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨 ERRO 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
        return false
    }
}


var knn = KNN(K: 5)


var data = readCSV(filename: "Iris")


var (trainData, testData) = getTrainAndTest(data: data, proportion: 0.2)
print("tamanho do train:", trainData.count)
print("tamanho do test:", testData.count)

knn.train(data: trainData)
var count: Float = 0.0
for singleTest in testData {
    if knn.predict(value: singleTest) == true {
        count += 1
    }
}
print(count)
print(testData.count)
print("A porcentagem de acertos é: \((count / Float(testData.count)) * 100)%")

