import Foundation
import UIKit
import CoreML
import Accelerate

//let labels = ["Chilsung-Cider", "Coca-Cola-Zero", "Fanta-Pineapple", "Gatorade", "Pocari-Sweat", "fingertip"]
let labels = ["칠성 사이다", "코카콜라 제로", "환타 파인애플", "게토레이드", "포카리스웨트", "fingertip"]
let products = labels[0...4]
let fingertipIndex = 5

let anchors: [Float] = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.

  Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

  - Parameters:
    - boxes: an array of bounding boxes and their scores
    - limit: the maximum number of boxes that will be selected
    - threshold: used to decide whether boxes overlap too much
*/
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {

  // Do an argsort on the confidence scores, from high to low.
  let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

  var selected: [YOLO.Prediction] = []
  var active = [Bool](repeating: true, count: boxes.count)
  var numActive = active.count
  var boxF = YOLO.Prediction(classIndex: fingertipIndex, score: 0, rect: CGRect(x: CGFloat(0), y: CGFloat(0), width: CGFloat(0), height: CGFloat(0))) //fingertipIndex --> fingertip 하나만 골라냄
    
  // The algorithm is simple: Start with the box that has the highest score.
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain
  // or the limit has been reached.
  outer: for i in 0..<boxes.count {
    if active[i] {
      let boxA = boxes[sortedIndices[i]]
      if boxA.classIndex == fingertipIndex {
         if boxA.score > boxF.score {
            boxF = boxA
         } //fingertipIndex --> fingertip 하나만 골라냄
      } else {selected.append(boxA)}
        
      if selected.count >= limit { break }

      for j in i+1..<boxes.count {
        if active[j] {
          let boxB = boxes[sortedIndices[j]]
          if IOU(a: boxA.rect, b: boxB.rect) > threshold {
            active[j] = false
            numActive -= 1
            if numActive <= 0 { break outer }
          }
        }
      }
    }
  }
  if boxF.score > 0 {selected.append(boxF)}
    
  return selected
}

/**
  Computes intersection-over-union overlap between two bounding boxes.
*/
public func IOU(a: CGRect, b: CGRect) -> Float {
  let areaA = a.width * a.height
  if areaA <= 0 { return 0 }

  let areaB = b.width * b.height
  if areaB <= 0 { return 0 }

  let intersectionMinX = max(a.minX, b.minX)
  let intersectionMinY = max(a.minY, b.minY)
  let intersectionMaxX = min(a.maxX, b.maxX)
  let intersectionMaxY = min(a.maxY, b.maxY)
  let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                         max(intersectionMaxX - intersectionMinX, 0)
  return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

extension Array where Element: Comparable {
  /**
    Returns the index and value of the largest element in the array.
  */
  public func argmax() -> (Int, Element) {
    precondition(self.count > 0)
    var maxIndex = 0
    var maxValue = self[0]
    for i in 1..<self.count {
      if self[i] > maxValue {
        maxValue = self[i]
        maxIndex = i
      }
    }
    return (maxIndex, maxValue)
  }
}

/**
  Logistic sigmoid.
*/
public func sigmoid(_ x: Float) -> Float {
  return 1 / (1 + exp(-x))
}

/**
  Computes the "softmax" function over an array.

  Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/

  This is what softmax looks like in "pseudocode" (actually using Python
  and numpy):

      x -= np.max(x)
      exp_scores = np.exp(x)
      softmax = exp_scores / np.sum(exp_scores)

  First we shift the values of x so that the highest value in the array is 0.
  This ensures numerical stability with the exponents, so they don't blow up.
*/
public func softmax(_ x: [Float]) -> [Float] {
  var x = x
  let len = vDSP_Length(x.count)

  // Find the maximum value in the input array.
  var max: Float = 0
  vDSP_maxv(x, 1, &max, len)

  // Subtract the maximum from all the elements in the array.
  // Now the highest value in the array is 0.
  max = -max
  vDSP_vsadd(x, 1, &max, &x, 1, len)

  // Exponentiate all the elements in the array.
  var count = Int32(x.count)
  vvexpf(&x, x, &count)

  // Compute the sum of all exponentiated values.
  var sum: Float = 0
  vDSP_sve(x, 1, &sum, len)

  // Divide each element by the sum. This normalizes the array contents
  // so that they all add up to 1.
  vDSP_vsdiv(x, 1, &sum, &x, 1, len)

  return x
}

func findDirection(fingertip : YOLO.Prediction, beverage : YOLO.Prediction) -> String {
  let fw = fingertip.rect.size.width
  let fh = fingertip.rect.size.height
  let fx = fingertip.rect.origin.x + fw/2 //fingertip center x
  let fy = fingertip.rect.origin.y + fh/2
  let bw = beverage.rect.size.width
  let bh = beverage.rect.size.height
  let bx = beverage.rect.origin.x //beverage topleft x
  let by = beverage.rect.origin.y
  var verticalDirection = ""
  var horizontalDirection = ""
  
  if fx < bx { verticalDirection = "right" }
  else if fx > bx+bw { verticalDirection = "left" }
  else { verticalDirection = "" } // bx < fx < bx+bw
  
  if fy < by { horizontalDirection = "down" }
  else if fy > by+bh { horizontalDirection = "up" }
  else { horizontalDirection = "" } // by < fy < by+h
  
  if verticalDirection == "", horizontalDirection == "" {
    return "correct"
  }
  else if verticalDirection == "", horizontalDirection != "" {
    return horizontalDirection
  }
  else if verticalDirection != "", horizontalDirection == "" {
    return verticalDirection
  }
  else {
    return verticalDirection + "_" + horizontalDirection
  }
}
