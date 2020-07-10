import os, sys, cv2, numpy as np
import torch, torchvision
from argparse import ArgumentParser
from recognition.model import RecognitionModel
from detection.unet import UNet
# from utils import prepare_for_inference, get_boxes_from_mask, four_point_transform, order_points
import utils
import segmentation_models_pytorch as smp


def ocr_preprocess(image, output_size):
    image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.
    return torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=str, default=None, help='path to the data')
    parser.add_argument('-t', '--seg_threshold', dest='seg_threshold', type=float, default=0.5,
                        help='decision threshold for segmentation model')
    parser.add_argument('-s', '--seg_model', dest='seg_model', type=str, default=None,
                        help='path to a trained segmentation model')
    parser.add_argument('-r', '--rec_model', dest='rec_model', type=str, default=None,
                        help='path to a trained recognition model')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='./submissions',
                        help='dir to save log and models')
    parser.add_argument('--input_wh', '-wh', dest='input_wh', type=str, help='model input size', default='320x32')
    parser.add_argument('--fit_size', '-fs', dest='fit_size', type=str, help='segmentation model fit size', default='256x256')

    args = parser.parse_args()
    print('Start inference')
    w, h = list(map(int, args.input_wh.split('x')))
    fit_size = list(map(int, args.fit_size.split('x')))

    cur_dir = os.path.dirname(__file__)
    seg_model_path = args.seg_model or os.path.join(cur_dir, 'pretrained', 'seg_unet_20epoch_basic.pth')
    rec_model_path = args.rec_model or os.path.join(cur_dir, 'pretrained', 'rec-crnn-30epoch_basic.pth')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seg_model = smp.FPN('se_resnet50', encoder_weights='imagenet', classes=2)
    seg_model.load_state_dict(torch.load(seg_model_path))
    seg_model.to(device)
    seg_model.eval()

    recognition_model = RecognitionModel(dropout=0.3, num_directions=2, input_size=(w, h))
    recognition_model.load_state_dict(torch.load(rec_model_path))
    recognition_model.to(device)
    recognition_model.eval()
    
    test_dir_path = os.path.join(args.data_path, 'test')
    results = []
    with torch.no_grad():
        files = os.listdir(test_dir_path)
        for i, file_name in enumerate(files):
            image_src = cv2.imread(os.path.join(test_dir_path, file_name))
            img, k, dw, dh = utils.prepare_for_inference(image_src.astype(np.float) / 255., (fit_size[0], fit_size[1]),) #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # TODO: obtain sizes from torch.load
            input = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
            
            #detection through segmentation
            pred = torch.sigmoid(seg_model(input.to(device))[:, 0]).squeeze().cpu().numpy()
            mask = (pred >= args.seg_threshold).astype(np.uint8)# * 255
            boxes = utils._get_boxes_from_mask(mask, dw, dh, k)
            if len(boxes) == 0:
                results.append((file_name, []))
                continue
                
            # crop corresponding bbox from the source image
            texts = []
            for box in boxes:
                box[:, 0] = box[:, 0].clip(0, image_src.shape[1] - 1)
                box[:, 1] = box[:, 1].clip(0, image_src.shape[0] - 1)
                x1 = box[0][0]
                crop = utils.crop_bounding_box(image_src, box)
                # OCR
                tensor = ocr_preprocess(crop, (w, h)).to(device) # don't forget to stay in sync with the augmentations
                text = recognition_model(tensor, decode=True)[0]
                texts.append((x1, text))
            
            # all predictions must be sorted by x1
            texts.sort(key=lambda x: x[0])
            results.append((file_name, [w[1] for w in texts]))
            if i % 100 == 0:
                print(i, len(files))
            
    # Generate a submission file
    with open(os.path.join(args.output_dir, 'submission.csv'), 'w') as wf:
        wf.write('file_name,plates_string\n')
        for file_name, texts in sorted(results, key=lambda x: int(os.path.splitext(x[0])[0])):
            wf.write('test/%s,%s\n' % (file_name, ' '.join(texts)))
    print('Done')
            

if __name__ == '__main__':
    sys.exit(main())
