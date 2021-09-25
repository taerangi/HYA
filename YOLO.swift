import Foundation
import UIKit
import CoreML

class YOLO {
  public static let inputWidth = 256
  public static let inputHeight = 256
  public static let maxBoundingBoxes = 10

  // Tweak these values to get more or fewer predictions.
  let confidenceThreshold: Float = 0.3
  let iouThreshold: Float = 0.5

  struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
  }
    
  let model = beta()
    
  public init() { }

  public func predict(image: CVPixelBuffer) -> [Prediction]? {
    if let output = try? model.prediction(image: image) {

      let list = [output._818, output._831, output._844]
//      let list = [output._714, output._727, output._740]
      var pred = [Prediction]()
      for i in list {
        if let k = list.firstIndex(of: i) {
          let cbb = computeBoundingBoxes(features: i, anchor_num: k) //return predictions
          pred += cbb
        }
      }
      
//      return computeBoundingBoxes(features: output._727) //714, 727, 740
//      return computeBoundingBoxes(features: output.grid)
//      print(nonMaxSuppression(boxes: pred, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold))
      return nonMaxSuppression(boxes: pred, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
        
    } else {
      return nil
    }
  }

  public func computeBoundingBoxes(features: MLMultiArray, anchor_num: Int) -> [Prediction] {
//  714 --> 1632000 = 85 * 80 * 80 * 3
//  727 --> 408000 = 85 * 40 * 40 * 3
//  740 --> 102000 = 85 * 20 * 20 * 3
//      grid * grid * (nClasses+5) * boxesPerCell
    
//    assert(features.count == 85*20*20*3)
//    assert(features.count == 125*13*13)  //[20(c)+5(xywhc)] * 5(boxesperCell) * 13(grid)
    
    var predictions = [Prediction]()

    let gridHeight = features.shape[2].intValue //20
    let gridWidth = features.shape[3].intValue //20
    let inputwidth : Int = 256 //640
    let blockSize: Float = Float(inputwidth / gridWidth) //32...
    let boxesPerCell = features.shape[1].intValue //3
    let numClasses = features.shape[4].intValue - 5 //80
    
//    let blockSize: Float = 32 //the number of pixels per grid cell : 640/20(grid) = 32
//    let gridHeight = 20
//    let gridWidth = 20
    
    
//    let gridHeight = 13
//    let gridWidth = 13
//    let boxesPerCell = 5
//    let numClasses = 20

    
//    all YOLOv5 models use 3 multi-scale outputs at strides 8, 16 and 32.
//    (by Netron)
//    output - type : float32[1,3,80,80,85] 714
//    417 - type : float32[1,3,40,40,85]    727
//    437 - type : float32[1,3,20,20,85]    740
//    numClasses : 80
    
    // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
    // five data items: x, y, width, height, and a confidence score. Each grid
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell
    // values for each grid cell, i.e. 125 channels. The total features array
    // contains 125x13x13 elements.

    // NOTE: It turns out that accessing the elements in the multi-array as
    // `features[[channel, cy, cx] as [NSNumber]].floatValue` is kinda(약간) slow.
    // It's much faster to use direct memory access to the features.
    
//    let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
    let featurePointer = UnsafeMutablePointer<Float>(OpaquePointer(features.dataPointer))

//    let channelStride = features.strides[0].intValue    //169 = 13*13?
//    let yStride = features.strides[1].intValue          //13
//    let xStride = features.strides[2].intValue          //1

//    [102000, 34000, 1700, 85, 1]
    let yStride = 1
    let xStride = gridHeight //20
    let bStride = gridHeight * gridWidth //400
    let channel = (numClasses+5) //85
//  The offset() helper function is used to find the proper place in the array to read from. Metal stores its data in texture slices in groups of 4 channels at a time, which means the 125 channels are not stored consecutively(연속하여) but are scattered all over the place. (See the code for an in-depth explanation.)
    
//    @inline(__always) func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
//      return channel*channelStride + y*yStride + x*xStride
//    }
    
    @inline(__always) func offset(_ x: Int, _ y: Int, _ b: Int) -> Int {
      return x*yStride + y*xStride + b*bStride
    }
    
    for cy in 0..<gridHeight {
      for cx in 0..<gridWidth {
        for b in 0..<boxesPerCell {

          // For the first bounding box (b=0) we have to read channels 0-24,
          // for b=1 we have to read channels 25-49, and so on.
//          let channel = b*(numClasses + 5) //[0...2]*85 = 0,85, 170
          
//          let count = (cy+1) * (cx+1) * (b+1) - 1 //0~1199  ...20*20*3
            
          // The slow way:
          /*
          let tx = features[[channel    , cy, cx] as [NSNumber]].floatValue
          let ty = features[[channel + 1, cy, cx] as [NSNumber]].floatValue
          let tw = features[[channel + 2, cy, cx] as [NSNumber]].floatValue
          let th = features[[channel + 3, cy, cx] as [NSNumber]].floatValue
          let tc = features[[channel + 4, cy, cx] as [NSNumber]].floatValue
          */
          
          // The fast way:
//          let tx = Float(featurePointer[offset(channel    , cx, cy)])
//          let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
//          let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
//          let th = Float(featurePointer[offset(channel + 3, cx, cy)])
//          let tc = Float(featurePointer[offset(channel + 4, cx, cy)]) //confidence

//          let tx = featurePointer[offset(cx, cy, b) * channel    ]
//          let ty = featurePointer[offset(cx, cy, b) * channel + 1]
//          let tw = featurePointer[offset(cx, cy, b) * channel + 2]
//          let th = featurePointer[offset(cx, cy, b) * channel + 3]
          let tc = featurePointer[offset(cx, cy, b) * channel + 4]

//          let tx = featurePointer[(channel * count)] // 85*[0...1199]
//          let ty = featurePointer[(channel * count) + 1]
//          let tw = featurePointer[(channel * count) + 2]
//          let th = featurePointer[(channel * count) + 3]
//          let tc = featurePointer[(channel * count) + 4]
            
          // The predicted tx and ty coordinates(좌표) are relative to the location
          // of the grid cell; we use the logistic sigmoid to constrain(제한하다) these
          // coordinates to the range 0 - 1. Then we add the cell coordinates
          // (0-12) and multiply by the number of pixels per grid cell (32). --> 32 = 640/20 = 416/13
          // Now x and y represent center of the bounding box in the original
          // 416x416 image space.
//          let x = (Float(cx) + sigmoid(tx)) * blockSize
//          let y = (Float(cy) + sigmoid(ty)) * blockSize

//          let x = (sigmoid(tx) * 2  - 0.5  + Float(cx)) * blockSize
//          let y = (sigmoid(ty) * 2  - 0.5  + Float(cy)) * blockSize
          //stride
          //shape : 1,3,80,80,85
            
          // The size of the bounding box, tw and th, is predicted relative to
          // the size of an "anchor" box. Here we also transform the width and
          // height into the original 416x416 image space.
          // Anchor 는 검출객체 너비, 높이의 초기값으로 주어진 값들이다. 이 초기값이 리사이즈되서 실제 검출객체 크기가 된다.

            // 56784 = 7 * 52 * 52 * 3
            // 14196 = 7 * 26 * 26 * 3
            // 3549 = 7 * 13 * 13 * 3
//          if features.count == 7 * 52 * 52 * 3 {
//                z += 6
//          }
//          if features.count == 7 * 26 * 26 * 3 {
//                z += 3
//          }
//          var anchor_count = 0
//          if features.count == 85 * 20 * 20 * 3 { //740
//            anchor_count += 6
//          }
//          if features.count == 85 * 40 * 40 * 3 { //727
//            anchor_count += 3
//          }
            
//          let w = exp(tw) * anchors[2*b    ] //* blockSize 나누기 2? 블록사이즈 고려해서 32는안나누고 16은 2 나누고 8 은 4나누고?
//          let h = exp(th) * anchors[2*b + 1] //* blockSize
            
//          let w = (sigmoid(tw)*2) * (sigmoid(tw)*2) * anchors[2*b     + 6*anchor_num]
//          let h = (sigmoid(th)*2) * (sigmoid(th)*2) * anchors[2*b + 1 + 6*anchor_num]
          // The confidence value for the bounding box is given by tc. We use
          // the logistic sigmoid to turn this into a percentage.
          let confidence = sigmoid(tc)

          // Gather the predicted classes for this anchor box and softmax them,
          // so we can interpret these numbers as percentages.
          var classes = [Float](repeating: 0, count: numClasses)
            
          for c in 0..<numClasses {
//            // The slow way:
////            classes[c] = features[[channel + 5 + c, cy, cx] as [NSNumber]].floatValue
//
//            // The fast way:
//            classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
//          }
            
//            classes[c] = featurePointer[(channel * count) + 5 + c]
            classes[c] = featurePointer[offset(cx, cy, b) * channel + 5 + c]
          }
          

          classes = softmax(classes)

          // Find the index of the class with the largest score.
          let (detectedClass, bestClassScore) = classes.argmax()

          // Combine the confidence score for the bounding box, which tells us
          // how likely it is that there is an object in this box (but not what
          // kind of object it is), with the largest class prediction, which
          // tells us what kind of object it detected (but not where).
          let confidenceInClass = bestClassScore * confidence

          // Since we compute 13x13x5 = 845 bounding boxes, we only want to
          // keep the ones whose combined score is over a certain threshold.
          if confidenceInClass > confidenceThreshold {
            let tx = featurePointer[offset(cx, cy, b) * channel    ]
            let ty = featurePointer[offset(cx, cy, b) * channel + 1]
            let tw = featurePointer[offset(cx, cy, b) * channel + 2]
            let th = featurePointer[offset(cx, cy, b) * channel + 3]
            let x = (sigmoid(tx) * 2  - 0.5  + Float(cx)) * blockSize
            let y = (sigmoid(ty) * 2  - 0.5  + Float(cy)) * blockSize
            let w = (sigmoid(tw)*2) * (sigmoid(tw)*2) * anchors[2*b     + 6*anchor_num]
            let h = (sigmoid(th)*2) * (sigmoid(th)*2) * anchors[2*b + 1 + 6*anchor_num]
            
            let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                              width: CGFloat(w), height: CGFloat(h))
//            if detectedClass == 16 , confidenceInClass > 0.4 {
//                print(b, tw, th, x, y, cx, cy)
//            }
//            if confidenceInClass > 0.3 {
//                print(b, detectedClass)
//            }
                
            let prediction = Prediction(classIndex: detectedClass,
                                        score: confidenceInClass,
                                        rect: rect)
            predictions.append(prediction)
            
//            print(predictions)
          }
        }
      }
    }

    // We already filtered out any bounding boxes that have very low scores,
    // but there still may be boxes that overlap too much with others. We'll
    // use "non-maximum suppression" to prune those duplicate bounding boxes.
//    print(nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold))
//    return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
    return predictions
  }
}
