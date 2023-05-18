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

//func getRotulos(from data:DataFrame) -> [String] {
//    var rotulos = data.summary(of: "Species")[row: 0]["mode", Array<Any>.self]
//    return rotulos as! [String]
//}

func getRotulos(from filename:String) -> [String] {
    var dataframe = readCSV(filename: filename)
    var rotulos = dataframe.summary(of: "Species")[row: 0]["mode", Array<Any>.self]
    return rotulos as! [String]
}


struct DMC {
    
    var centroids: [Iris] = []
    
    mutating func train(data: [Iris]) {
        var rotulos = getRotulos(from: "Iris")
        print(rotulos)
        for rotulo in rotulos {
            var centroidIris = Iris(atributo: SIMD4<Float>(), rotulo: rotulo)
            var dataForCentroid = data.filter({$0.rotulo == rotulo})
            
            for data in dataForCentroid {
                centroidIris.atributo += data.atributo
            }
            centroidIris.atributo /= Float(dataForCentroid.count)
            self.centroids.append(centroidIris)
            
        }
        
        
    }
    func predict(value: Iris) -> Bool {
        var distances = [Float:String]()
        for centroid in centroids {
            let dist = distance(centroid.atributo, value.atributo)
            distances[dist] = centroid.rotulo
        }
        let sortedKeys = Array(distances.keys).sorted(by: <)
        print(sortedKeys)
        
        return distances[sortedKeys[0]] == value.rotulo
        
        
    }
    
}

var dmc = DMC()
var dataFrame = readCSV(filename: "Iris")
var (train, test) = getTrainAndTest(data: dataFrame, proportion: 0.5)

dmc.train(data: train)
var count = 0
for testData in test {
    if dmc.predict(value: testData) {
        count += 1
    }
}

print("A porcentagem de acerto é: \(Float(count) / Float(test.count) * 100)")
