from src.algorithm import *
from src.network import *
from argparse import ArgumentParser

BOOST = True

label2piece = {"0":"b","1":"k","2":"n","3":"p","4":"q","5":"r","6":"B","7":"K","8":"N","9":"P","10":"Q","11":"R","12":"z"}

def most_frequent(List): 
    return max(set(List), key = List.count) 

def make_args():
    parser = ArgumentParser()

    #dataset 
    parser.add_argument('--image_path', dest='image_path', type=str, help='image file path', required=True)
    parser.add_argument('--model_path', dest="model_path", default="2_5_2021.pth", type=str, help='CNN model path')
    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    img = load_image(args.image_path) #examples/{}.jpg
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(.001)

    edges = get_contours(img)
    perspective = get_perspective_from_contours(edges[0])
    board =get_boards_from_perspective(img, perspective)

    model = ConvolutionalNeuralNetwork(1, 13)
    model.load_state_dict(torch.load("2_5_2021.pth"))

    model.eval()

    my_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])


    # Get FEN
    fen_list = []
    for i in range(8):
        image_list = []
        for j in range(8):
            imgk = PIL.Image.fromarray(board.getTile(i,j).getImage())
            imgk = imgk.convert("L")
            for k in range(4):
                imgk.rotate(90 * k)
                imgk_x = my_transforms(imgk)
                image_list.append(imgk_x)

        image_tensor = torch.stack(image_list, dim=0)

        outputs = model(image_tensor)

        labels = torch.argmax(outputs, dim = 1).data.numpy()

        if BOOST:
            labels = list(labels)
            labels = [most_frequent(labels[i * 4:(i+1) * 4]) for i in range(8)]
        
        fen_line = []
        for j, label in enumerate(labels):
            piece = label2piece[str(label)]
            if piece == "z":
                if j >= 1 and fen_line[-1].isdigit():
                    code = str(int(fen_line[-1]) + 1)
                    fen_line[-1] = code
                else:
                    fen_line.append("1")
            else:
                fen_line.append(piece)

        fen_line_str = "".join(fen_line)
        fen_list.append(fen_line_str)

    
    fen = "/".join(fen_list)
    print(fen)

        
    os.system('pause')