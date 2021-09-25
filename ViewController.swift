import UIKit
import Vision
import AVFoundation
import CoreMedia

class ViewController: UIViewController {
  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var timeLabel: UILabel!
//  @IBOutlet weak var debugImageView: UIImageView!
  @IBOutlet weak var help: UILabel!
  @IBOutlet weak var productNameField: UITextField!
  @IBOutlet weak var searchButton: UIButton!
  @IBOutlet weak var productList: UILabel!
  
  var productNamePick = UIPickerView()
    
  var detectLocationFlag = false
    
  @IBAction func didTapButton() {
    if detectLocationFlag == false {
        detectLocationFlag = true
        searchButton.setTitle("Stop", for: .normal)
    }
    else {
        detectLocationFlag = false
        searchButton.setTitle("Search", for: .normal)
    }
  }
    
  var soundEffect: AVAudioPlayer?
  var soundPath = Bundle.main.path(forResource: "correct", ofType:"mp3")!
  var soundInterval = 1
  var intervalCount = 0

  // true: use Vision to drive Core ML, false: use plain Core ML
  let useVision = false

  // Disable this to see the energy impact of just running the neural net,
  // otherwise it also counts the GPU activity of drawing the bounding boxes.
  let drawBoundingBoxes = true

  // How many predictions we can do concurrently.
  static let maxInflightBuffers = 3

  let yolo = YOLO()

  var videoCapture: VideoCapture!
  var requests = [VNCoreMLRequest]()
  var startTimes: [CFTimeInterval] = []

  var boundingBoxes = [BoundingBox]()
  var colors: [UIColor] = []

  let ciContext = CIContext()
  var resizedPixelBuffers: [CVPixelBuffer?] = []

  var framesDone = 0
  var frameCapturingStartTime = CACurrentMediaTime()

  var inflightBuffer = 0
  let semaphore = DispatchSemaphore(value: ViewController.maxInflightBuffers)

  override func viewDidLoad() {
    super.viewDidLoad()

    timeLabel.text = ""
    productNameField.delegate = self
    productNamePick.delegate = self
    productNamePick.dataSource = self
    productNameField.inputView = productNamePick
    dismissPickerView()
    
//    let firstLaunch = FirstLaunch(userDefaults: .standard, key: "com.any-suggestion.FirstLaunch.WasLaunchedBefore")
//    if firstLaunch.isFirstLaunch {
//        location.text = "first launch"
//        // do things
//    }

    setUpBoundingBoxes()
    setUpCoreImage()
    //setUpVision()
    setUpCamera()

    frameCapturingStartTime = CACurrentMediaTime()
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }

  // MARK: - Initialization

  func setUpBoundingBoxes() {
    for _ in 0..<YOLO.maxBoundingBoxes {
      boundingBoxes.append(BoundingBox())
    }

    // Make colors for the bounding boxes. There is one color for each class,
    // 20 classes in total.
    for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
      for g: CGFloat in [0.3, 0.5, 0.7, 0.9] {
        for b: CGFloat in [0.2, 0.4, 0.6, 0.8] {
            let color = UIColor(red: r, green: g, blue: b, alpha: 1)
          colors.append(color)
        }
      }
    }
  }

  func setUpCoreImage() {
    // Since we might be running several requests in parallel, we also need
    // to do the resizing in different pixel buffers or we might overwrite a
    // pixel buffer that's already in use.
    for _ in 0..<ViewController.maxInflightBuffers {
      var resizedPixelBuffer: CVPixelBuffer?
      let status = CVPixelBufferCreate(nil, YOLO.inputWidth, YOLO.inputHeight,
                                       kCVPixelFormatType_32BGRA, nil,
                                       &resizedPixelBuffer)

      if status != kCVReturnSuccess {
        print("Error: could not create resized pixel buffer", status)
      }
      resizedPixelBuffers.append(resizedPixelBuffer)
    }
  }

//  func setUpVision() {
//    guard let visionModel = try? VNCoreMLModel(for: yolo.model.model) else {
//      print("Error: could not create Vision model")
//      return
//    }
//
//    for _ in 0..<ViewController.maxInflightBuffers {
//      let request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
//
//      // NOTE: If you choose another crop/scale option, then you must also
//      // change how the BoundingBox objects get scaled when they are drawn.
//      // Currently they assume the full input image is used.
//      request.imageCropAndScaleOption = .scaleFill
//      requests.append(request)
//    }
//  }

  func setUpCamera() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self
    videoCapture.desiredFrameRate = 60 //240
    videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.hd1280x720) { success in
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.resizePreviewLayer()
        }

        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxes {
          box.addToLayer(self.videoPreview.layer)
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  // MARK: - UI stuff

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  func resizePreviewLayer() {
    videoCapture.previewLayer?.frame = videoPreview.bounds
  }

    // MARK: - Doing inference


  func predict(image: UIImage) {
    if let pixelBuffer = image.pixelBuffer(width: YOLO.inputWidth, height: YOLO.inputHeight) {
      predict(pixelBuffer: pixelBuffer, inflightIndex: 0)
    }
  }

  func predict(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
    // Measure how long it takes to predict a single video frame.
    let startTime = CACurrentMediaTime()

    // This is an alternative way to resize the image (using vImage):
    //if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
    //                                              width: YOLO.inputWidth,
    //                                              height: YOLO.inputHeight) {

    // Resize the input with Core Image to 416x416.
    if let resizedPixelBuffer = resizedPixelBuffers[inflightIndex] {
      let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
      let sx = CGFloat(YOLO.inputWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
      let sy = CGFloat(YOLO.inputHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
      let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
      let scaledImage = ciImage.transformed(by: scaleTransform)
      ciContext.render(scaledImage, to: resizedPixelBuffer)

      // Give the resized input to our model.
      if let boundingBoxes = yolo.predict(image: resizedPixelBuffer) {
        let elapsed = CACurrentMediaTime() - startTime
        showOnMainThread(boundingBoxes, elapsed)
      } else {
        print("BOGUS")
      }
    }

    self.semaphore.signal()
  }

  func predictUsingVision(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
    // Measure how long it takes to predict a single video frame. Note that
    // predict() can be called on the next frame while the previous one is
    // still being processed. Hence the need to queue up the start times.
    startTimes.append(CACurrentMediaTime())

    // Vision will automatically resize the input image.
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
    let request = requests[inflightIndex]

    // Because perform() will block until after the request completes, we
    // run it on a concurrent background queue, so that the next frame can
    // be scheduled in parallel with this one.
    DispatchQueue.global().async {
      try? handler.perform([request])
    }
  }

  func visionRequestDidComplete(request: VNRequest, error: Error?) {
    if let observations = request.results as? [VNCoreMLFeatureValueObservation],
       let features = observations.first?.featureValue.multiArrayValue {
//       let features = observations.last?.featureValue.multiArrayValue {
      
      let boundingBoxes = yolo.computeBoundingBoxes(features: features, anchor_num: 2)
      let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)
      showOnMainThread(boundingBoxes, elapsed)
    } else {
      print("BOGUS!")
    }
    

    self.semaphore.signal()
  }

    func soundPlay(predictions : [YOLO.Prediction], target : String) {
      let productIndex = labels.firstIndex(of: target) // 찾고자 하는 제품의 index
      
      // 1. 화면 상에 찾고자 하는 제품이 존재할 때
      if let beverage = predictions.first(where: {$0.classIndex == productIndex}) {
        
        // 1-1. 화면 상에 손가락이 존재할 때
        if let fingertip = predictions.first(where: {$0.classIndex == fingertipIndex}) {
          let direction = findDirection(fingertip: fingertip, beverage: beverage)
          
          // 오디오 파일 재생이 겹치는 것을 방지하기 위해 간격(interval)을 설정
          // soundInterval이 1이면 오디오 파일 재생
          // soundInterval이 0이면 intervalCount를 하나씩 늘리고, 일정 수준을 넘어서면 soundInterval을 1로 변경
          if self.soundInterval == 1 {
            self.soundPath = Bundle.main.path(forResource: direction, ofType:"mp3")!
            do {
              self.soundEffect = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: self.soundPath))
              self.soundEffect?.play()
                
              // soundInterval, intervalCount 초기화
              self.soundInterval = 0
              self.intervalCount = 0
            } catch {}
          }
          else {
            self.intervalCount += 1
            if self.intervalCount >= 35 { self.soundInterval = 1 }
          }
        }
        
        // 1-2. 화면 상에 손가락이 존재하지 않을 때
        else if self.soundInterval == 1 {
          do {
            self.soundPath = Bundle.main.path(forResource: "show_indexfinger", ofType:"mp3")!
            self.soundEffect = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: self.soundPath))
            self.soundEffect?.play()
            self.soundInterval = 0
            self.intervalCount = 0
          } catch {}
        }
        else {
          self.intervalCount += 1
          if self.intervalCount >= 200 { self.soundInterval = 1 }
        }
      }
      
      // 2. 찾고자 하는 제품이 존재하지 않을 때
      else if self.soundInterval == 1 {
        do {
          self.soundPath = Bundle.main.path(forResource: "no_product", ofType:"mp3")!
          self.soundEffect = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: self.soundPath))
          self.soundEffect?.play()
          self.soundInterval = 0
          self.intervalCount = 0
        } catch {}
      }
      else {
        self.intervalCount += 1
        if self.intervalCount >= 120 { self.soundInterval = 1 }
      }
  }
    
  func showOnMainThread(_ boundingBoxes: [YOLO.Prediction], _ elapsed: CFTimeInterval) {
    if drawBoundingBoxes {
      DispatchQueue.main.async {
        // For debugging, to make sure the resized CVPixelBuffer is correct.
        //var debugImage: CGImage?
        //VTCreateCGImageFromCVPixelBuffer(resizedPixelBuffer, nil, &debugImage)
        //self.debugImageView.image = UIImage(cgImage: debugImage!)

        self.show(predictions: boundingBoxes)
        
        // productNameField에서 원하는 제품을 선택한 후 Search 버튼을 눌렀을 때, soundPlay 함수를 실행
        if let productName = self.productNameField.text, self.detectLocationFlag == true {
            self.soundPlay(predictions: boundingBoxes, target: productName)
        }
        /*
        if self.detectLocationFlag == true, let productName = self.productNameField.text {
           let productIndex = labels.firstIndex(of: productName)
           if let beverage = boundingBoxes.first(where: {$0.classIndex == productIndex}) {
                if let fingertip = boundingBoxes.first(where: {$0.classIndex == fingertipIndex}) {
                    let direction = findDirection(fingertip: fingertip, beverage: beverage)
//                    self.location.text = String(format: direction)
                    if self.soundInterval == 0 { //, self.intervalCount != 50
                        self.soundPath = Bundle.main.path(forResource: direction, ofType:"mp3")!
                        do {
                            self.soundEffect = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: self.soundPath))
                            self.soundEffect?.play()
                            self.soundInterval = 1
                            self.intervalCount = 0
                        } catch {}
                    }
                    else {
                        self.intervalCount += 1
                        if self.intervalCount >= 35 { self.soundInterval = 0 }
                    }
                }
                else if self.soundInterval == 0 {
                    do {
                        self.soundPath = Bundle.main.path(forResource: "show_indexfinger", ofType:"mp3")!
                        self.soundEffect = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: self.soundPath))
                        self.soundEffect?.play()
                        self.soundInterval = 1
                        self.intervalCount = 0
                    } catch {}
                    //self.location.text = String(format: "Please show your index finger on the screen to find the exact location of the product")
                }
                else {
                    self.intervalCount += 1
                    if self.intervalCount >= 200 { self.soundInterval = 0 }
                }
           }
           else if self.soundInterval == 0 {
            do {
                self.soundPath = Bundle.main.path(forResource: "no_product", ofType:"mp3")!
                self.soundEffect = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: self.soundPath))
                self.soundEffect?.play()
                self.soundInterval = 1
                self.intervalCount = 0
            } catch {}
            //self.location.text = String(format: "There's no product you're looking for")
           }
           else {
            self.intervalCount += 1
            if self.intervalCount >= 120 { self.soundInterval = 0 }
           }
        }
        */
        let fps = self.measureFPS()
        self.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)
      }
    }
  }

  func measureFPS() -> Double {
    // Measure how many frames were actually delivered per second.
    framesDone += 1
    let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
    let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
    if frameCapturingElapsed > 1 {
      framesDone = 0
      frameCapturingStartTime = CACurrentMediaTime()
    }
    return currentFPSDelivered
  }

  func show(predictions: [YOLO.Prediction]) {
    var classList : [String] = []
    for i in 0..<boundingBoxes.count {
      if i < predictions.count {
        let prediction = predictions[i]
        
        let className = labels[prediction.classIndex]
        if classList.contains(className) == false, className != "fingertip" {
            classList.append(className)
        }
        // The predicted bounding box is in the coordinate space of the input
        // image, which is a square image of 416x416 pixels. We want to show it
        // on the video preview, which is as wide as the screen and has a 16:9
        // aspect ratio. The video preview also may be letterboxed at the top
        // and bottom.
        let width = view.bounds.width
        let height = width * 16 / 9
        let scaleX = width / CGFloat(YOLO.inputWidth)
        let scaleY = height / CGFloat(YOLO.inputHeight)
        let top = (view.bounds.height - height) / 2

        // Translate and scale the rectangle to our own coordinate system.
        var rect = prediction.rect
        rect.origin.x *= scaleX
        rect.origin.y *= scaleY
        rect.origin.y += top
        rect.size.width *= scaleX
        rect.size.height *= scaleY

        // Show the bounding box.
        let label = String(format: "%@ %.1f", labels[prediction.classIndex], prediction.score * 100)
        let color = colors[prediction.classIndex]
        
        boundingBoxes[i].show(frame: rect, label: label, color: color)
      } else {
        boundingBoxes[i].hide()
      }
    classList.sort()
    let classListString = classList.joined(separator: ", ")
    if classListString == "" {
        self.productList.text = "제품이 없습니다"
    }
    else { self.productList.text = classListString }
    }
  }
}

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
    // For debugging.
//    return predict(image: UIImage(named: "dog640")!);

    if let pixelBuffer = pixelBuffer {
      // The semaphore will block the capture queue and drop frames when
      // Core ML can't keep up with the camera.
      semaphore.wait()

      // For better throughput, we want to schedule multiple prediction requests
      // in parallel. These need to be separate instances, and inflightBuffer is
      // the index of the current request.
      let inflightIndex = inflightBuffer
      inflightBuffer += 1
      if inflightBuffer >= ViewController.maxInflightBuffers {
        inflightBuffer = 0
      }

      if useVision {
        // This method should always be called from the same thread!
        // Ain't nobody likes race conditions and crashes.
        self.predictUsingVision(pixelBuffer: pixelBuffer, inflightIndex: inflightIndex)
      } else {
        // For better throughput, perform the prediction on a concurrent
        // background queue instead of on the serial VideoCapture queue.
        DispatchQueue.global().async {
          self.predict(pixelBuffer: pixelBuffer, inflightIndex: inflightIndex)
        }
      }
    }
  }
}

extension ViewController : UITextFieldDelegate {
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        return true
    }
}

extension ViewController : UIPickerViewDelegate, UIPickerViewDataSource {
    func numberOfComponents(in pickerView : UIPickerView) -> Int {
        return 1 // 하나의 PickerView 안에 몇 개의 선택 가능한 리스트를 표시할 것인지
    }
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return products.count //PickerView에 표시될 항목의 개수
    }
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return products[row] //PickerView 내에서 특정한 위치(row)를 가리키게 될 때, 해당하는 문자열을 반환
    }
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        productNameField.text = products[row]
//        productNameField.resignFirstResponder() //picker 종료
    }
    func dismissPickerView() {
        let toolBar = UIToolbar()
        toolBar.sizeToFit()
        let button = UIBarButtonItem(title: "선택", style: .plain, target: self, action: #selector(pickerExit))
        let spaceButton = UIBarButtonItem(barButtonSystemItem: UIBarButtonItem.SystemItem.flexibleSpace, target: nil, action: nil)
        toolBar.setItems([spaceButton, spaceButton, button], animated: true)
        toolBar.isUserInteractionEnabled = true
        productNameField.inputAccessoryView = toolBar
    }
    @objc func pickerExit() {
        /// picker와 같은 뷰를 닫는 함수
        self.view.endEditing(true)
    }
}

//final class FirstLaunch {
//    let wasLaunchedBefore: Bool
//    var isFirstLaunch: Bool {
//        return !wasLaunchedBefore
//    }
//    
//    init(getWasLaunchedBefore: () -> Bool, setWasLaunchedBefore: (Bool) -> ()) {
//        let wasLaunchedBefore = getWasLaunchedBefore()
//        self.wasLaunchedBefore = wasLaunchedBefore
//        if !wasLaunchedBefore { setWasLaunchedBefore(true) } }
//    
//    convenience init(userDefaults: UserDefaults, key: String) {
//        self.init(getWasLaunchedBefore: { userDefaults.bool(forKey: key) }, setWasLaunchedBefore: { userDefaults.set($0, forKey: key) }) }
//}
