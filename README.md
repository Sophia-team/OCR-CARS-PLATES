# OCR-CARS-PLATES
Detection and recognition of cars' numbers

## Detection Part
Detection is done with Feature Pyramid Network with se-resnet50 backbone pretrained on ImageNet

## Recognition Part
Resnet34 was selected for extraction features from car plate number image. Then simple Recurrent NN (GRU) was applied for text generation.
