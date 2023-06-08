

# conda环境：centernet  ==>   conda activate centernet


from super_gradients.training import models  
from super_gradients.common.object_names import Models  
  
net = models.get(Models.YOLO_NAS_S, pretrained_weights='coco')  
net.predict("input/bus.jpg").show()  


models.convert_to_onnx(model=net, input_shape=(3, 640, 640), out_path='yolo-nas-s.onnx') 





