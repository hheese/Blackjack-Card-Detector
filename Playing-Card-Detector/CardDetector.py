import cv2 as cv
import numpy as np
import os


# declare constants:
# threshold levels
BKG_THRESH = 150 
CARD_THRESH = 50
# width and height of card corner
CORNER_WIDTH = 32
CORNER_HEIGHT = 84
# dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125
# dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100
# max difference for identification
RANK_DIFF_MAX = 3000
SUIT_DIFF_MAX = 1700
# max/min area for contour detection
CARD_MAX_AREA = 2500000
CARD_MIN_AREA = 25000
FONT = cv.FONT_HERSHEY_SIMPLEX


# Card structures modified from https://hackaday.io/project/27639-rain-man-20-blackjack-robot
class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.rank_img = [] # Thresholded, sized image of card's rank
        self.suit_img = [] # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown" # Best matched rank
        self.best_suit_match = "Unknown" # Best matched suit
        self.rank_diff = 0 # Difference between rank image and best matched train rank image
        self.suit_diff = 0 # Difference between suit image and best matched train suit image

class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = [] # Thresholded, sized rank image loaded from hard drive
        self.name = "Placeholder"

class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = [] # Thresholded, sized suit image loaded from hard drive
        self.name = "Placeholder"

# Training images '/Card_Imgs' and training image loading modified from https://hackaday.io/project/27639-rain-man-20-blackjack-robot
def load_ranks(filepath):

    train_ranks = []
    i = 0
    
    for Rank in ['Ace','2','3','4','5','6','7',
                 '8','9','10','Jack','Queen','King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv.imread(filepath+filename, cv.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks

def load_suits(filepath):
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""

    train_suits = []
    i = 0
    
    for Suit in ['Spades','Diamonds','Clubs','Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv.imread(filepath+filename, cv.IMREAD_GRAYSCALE)
        i = i + 1

    return train_suits


def preprocess_image(image):
    # converts to grayscale, blurs, then binary thresholds

    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)
    _, thresh = cv.threshold(blur,BKG_THRESH,255,cv.THRESH_BINARY)

    return thresh

def find_cards(thresh_image):
    # find contours and sort indices by size
    contours,hier = cv.findContours(thresh_image,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(contours)), key=lambda i : cv.contourArea(contours[i]),reverse=True)

    if len(contours) == 0:
        return [], []
    
    contour_sort = []
    hier_sort = []
    contour_is_card = np.zeros(len(contours),dtype=int)

    for i in index_sort:
        contour_sort.append(contours[i])
        hier_sort.append(hier[0][i])

    # determine cards based on: area between max and min, no parent contours, has four corners
    for i in range(len(contour_sort)):
        size = cv.contourArea(contour_sort[i])
        peri = cv.arcLength(contour_sort[i],True)
        approx = cv.approxPolyDP(contour_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            contour_is_card[i] = 1

    return contour_sort, contour_is_card

# Card preprocessing modified from https://hackaday.io/project/27639-rain-man-20-blackjack-robot
def preprocess_card(contour, image):
    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv.arcLength(contour,True)
    approx = cv.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Find center point of card by taking x and y average of the four corners.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened image using perspective transform
    qCard.warp = flattener(image, pts, w, h)

    # Grab corner of warped card image and do a 4x zoom
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv.resize(Qcorner, (0,0), fx=4, fy=4)

    # Sample known white pixel intensity to determine good threshold level
    white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    _, query_thresh = cv.threshold(Qcorner_zoom, thresh_level, 255, cv.THRESH_BINARY_INV)

    # Split in to top and bottom half (top shows rank, bottom shows suit)
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # Find rank contour and bounding rectangle, isolate and find largest contour
    Qrank_cnts, hier = cv.findContours(Qrank, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv.contourArea,reverse=True)

    # Find bounding rectangle for largest contour, use it to resize query rank
    # image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # Find suit contour and bounding rectangle, isolate and find largest contour
    Qsuit_cnts, _ = cv.findContours(Qsuit, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv.contourArea,reverse=True)
    
    # Find bounding rectangle for largest contour, use it to resize query suit
    # image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        x2,y2,w2,h2 = cv.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard

def match_card(qCard, train_ranks, train_suits):
    # finds best match for card suit and rank based on absolute difference of training images
    best_rank_diff = 10000
    best_suit_diff = 10000
    best_rank = "Unknown"
    best_suit = "Unknown"
    i = 0

    # If no contours were found in preprocess_card function, skip process
    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):
        
        for rank in train_ranks:
                diff_img = cv.absdiff(qCard.rank_img, rank.img)
                rank_diff = int(np.sum(diff_img)/255)
                
                if (rank_diff < RANK_DIFF_MAX):
                    if rank_diff < best_rank_diff:
                        best_rank_diff = rank_diff
                        best_rank = rank.name

        for suit in train_suits:
                diff_img = cv.absdiff(qCard.suit_img, suit.img)
                suit_diff = int(np.sum(diff_img)/255)
                
                if (suit_diff < SUIT_DIFF_MAX):
                    if suit_diff < best_suit_diff:
                        best_suit_diff = suit_diff
                        best_suit = suit.name

    # returns card info and match quality
    return best_rank, best_suit, best_rank_diff, best_suit_diff
    
    
def draw_results(image, qCard):
    # draw the center mark and card info
    x = qCard.center[0]
    y = qCard.center[1]
    cv.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    # draw rank and suit
    cv.putText(image,(rank_name+' of'),(x-60,y-35),FONT,3,(0,0,0),12,cv.LINE_AA)
    cv.putText(image,(rank_name+' of'),(x-60,y-35),FONT,3,(255,255,255),8,cv.LINE_AA)

    cv.putText(image,suit_name,(x-60,y+45),FONT,3,(0,0,0),12,cv.LINE_AA)
    cv.putText(image,suit_name,(x-60,y+45),FONT,3,(255,255,255),8,cv.LINE_AA)
    

    return image

def flattener(image, pts, w, h):
    # Flattens an image of a card into a top-down 200x300 perspective.
    # Function used as is from www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
    maxWidth = 200
    maxHeight = 300
    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv.getPerspectiveTransform(temp_rect,dst)
    warp = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv.cvtColor(warp,cv.COLOR_BGR2GRAY)

    return warp

def main():
    # load the train rank and suit images
    train_ranks = load_ranks('Card_Imgs/')
    train_suits = load_suits('Card_Imgs/')


    inpath = ('../test_images/img/')
    outpath = ('../outputs/')
    #image = cv2.imread( path + '/input.png')

    imagecount= cardcount= suitcount= rankcount= ucountsuit= ucountrank= 0
    for filename in os.listdir(inpath):
        # skip over images that are out of scope: joker cards and complex backgrounds
        if filename[8] == "W": continue
        if filename[13] == "3": continue
        imagecount += 1

        image = cv.imread(inpath+'/'+filename)

        # grayscale, blur, binary threshold image
        pre_proc = preprocess_image(image)

        # find/sort contours
        contours_sort, contour_is_card = find_cards(pre_proc)

        if len(contours_sort) != 0:
            cards = []
            k = 0
            # for each card contour
            for i in range(len(contours_sort)):
                if (contour_is_card[i]):

                    # find corners, perspective warps card, isolates card info from top corner
                    cards.append(preprocess_card(contours_sort[i],image))

                    # find best match for card rank and suit
                    cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = match_card(cards[k],train_ranks,train_suits)

                    # compare results
                    image = draw_results(image, cards[k])
                    suit = cards[k].best_suit_match
                    rank = cards[k].best_rank_match
                    print("filename:", filename)
                    print("Expected:", filename[7], filename[8])
                    print("Result:  ", suit[0], rank[0])
                    if suit[0] == "U": ucountsuit+=1
                    if rank[0] == "U": ucountrank+=1
                    if filename[7]==suit[0]: suitcount+=1
                    if filename[8]==rank[0]: rankcount+=1
                    elif(filename[8]=="0" and rank[0]=="1"): rankcount+=1
                    cardcount += 1
                    k += 1
                    
                
            # draw contours on image 
            if (len(cards) != 0):
                temp_cnts = []
                for i in range(len(cards)):
                    temp_cnts.append(cards[i].contour)
                cv.drawContours(image,temp_cnts, -1, (255,0,0), 10)
                cv.imwrite( outpath + "/" + filename, image)
            
    print("Images Processed:",imagecount)
    print("Images with Card Detected:",cardcount, '/', imagecount)
    print("Suits Correctly Identified:",suitcount, '/', cardcount)
    print("Ranks Correctly Identified:",rankcount, '/', cardcount)
    cv.destroyAllWindows()

main()