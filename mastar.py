import astar
import cv2
from queue import PriorityQueue
import numpy as np
from threading import Timer

NODE_Obstacle = "obstacle"
NODE_Free = "free"
NODE_InOpenList = 'open'
NODE_InCloseList = 'close'
NODE_Unknown = 4

adjacent=[]
adjacent.extend([(1,0),(1,1),(1,-1),(0,1),(0,-1),(-1,0),(-1,1),(-1,-1)])
print(adjacent)
Inf = float('inf')
class Node(object):
    def __init__(self,x=0,y=0,g=Inf,f=Inf,next_=None):
        self.x = x
        self.y = y
        self.h = 0
        self.g = g
        self.f = f
        self.status = NODE_Free
        self.ParentNode=next_
    def point(self):
        return (self.x,self.y)

    def __lt__(self,other):
        if self.f < other.f:
            return True
        else:
            return False
    def __repr__(self):
        return str((self.x,self.y,self.f,self.status))

class mAstar(object):
    def __init__(self):
        self.path=[]
        self.panel = "Img"
        self.img=None
        self.graph={}
        self.start=None
        self.end=None
        self.graph=None
    
    def refresh(self):
        self.show()
        self.mTimer = Timer(20,self.refresh)
        self.mTimer.start()

    def load_img(self,imgpath):
        self.img = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
        self.img_raw = self.img.copy()

    def mousecallback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONUP:
            if self.start!=None:
                self.end = Node(x,y)
                cv2.circle(self.img,(x,y),1,0,1)
                cv2.imshow(self.panel,self.img)
                self.search(self.start,self.end)
                self.start=None
                return 
            self.img = self.img_raw.copy()
            self.start = Node(x,y)
            cv2.circle(self.img,(x,y),1,0,1)
            cv2.imshow(self.panel,self.img)

    def show(self):
        cv2.namedWindow(self.panel,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.panel,self.mousecallback)
        cv2.imshow(self.panel,self.img)
        cv2.waitKey()

    def search(self,start,end):
        self.img[start.y][start.x]=50
        self.img[end.y][end.x]=50
        openlist = PriorityQueue()
        closelist= []
        width = self.img.shape[1]
        height=self.img.shape[0]
        
        # input()
        t = cv2.getTickCount()
        print(t)
        for x in range(0,width):
            for y in range(0,height):
                tnode = Node(x,y)
                if self.img[y,x]>250:
                    tnode.status = NODE_Free
                else:
                    tnode.status= NODE_Obstacle
                    # print('obstacle',tnode)
                if x not in self.graph:
                    self.graph[x]={}    
                self.graph[x][y]=tnode
        print(cv2.getTickCount()-t)
        tnode = Node(start.x,start.y,0,0)
        tnode.status = NODE_Free
        self.graph[start.x][start.y]=tnode
        tnode = Node(end.x,end.y)
        tnode.status = NODE_Free
        self.graph[end.x][end.y]=tnode

        openlist.put(self.graph[start.x][start.y])

        t=cv2.getTickCount()
        print('start:',t)
        while not openlist.empty():
            # input()
            current = openlist.get()
            # print('\nclose:',current)
            if current.x == end.x and current.y == end.y:
                end=current
                print("Found.\n")

                break
            current.status = NODE_InCloseList
            self.img[current.y,current.x]=128
            self.img[start.y][start.x]=50
            for i in range(0,8):
                # input()
                xnow = current.x + adjacent[i][0]
                ynow = current.y + adjacent[i][1]
                if 0<=xnow<width and 0<=ynow<height:
                    neighbor = self.graph[xnow][ynow]
                    if neighbor.status == NODE_Obstacle or neighbor.status == NODE_InCloseList:
                        # print('ignore:',neighbor)
                        continue
                    cost = self.movementcost(current,neighbor) + current.g
                    if cost >= neighbor.g:
                        # print('cost {0} bigger than {1}:'.format(cost,neighbor.g))
                        continue
                    neighbor.ParentNode = current
                    neighbor.g = cost
                    neighbor.f = neighbor.g+self.heuristic(neighbor,end)
                    # print("put:",neighbor)
                    openlist.put(neighbor)
                    # cv2.namedWindow(self.panel,cv2.WINDOW_NORMAL)
            # cv2.imshow(self.panel,self.img)

        print('finished')
        node=end.ParentNode
        print(node)
        print(cv2.getTickCount()-t)
        while node.x!=start.x or node.y!=start.y:
            cv2.circle(self.img,node.point(),1,0,1)
            self.img[node.y][node.x]=80
            node = node.ParentNode
            print(node)
            cv2.imshow(self.panel,self.img)
            cv2.waitKey(1)
        cv2.waitKey(2)


    def movementcost(self,A,B):
        stepX = abs(A.x - B.x)
        stepY = abs(A.y - B.y)
        step = stepX + stepY
        if step == 1:
            cost = 10
        else:
            cost = 14
        return cost

    def heuristic(self,A,B):
        h = 10*((abs(A.x-B.x)+abs(A.y-B.y)))
        return h


if __name__ == "__main__":

    pf=mAstar()
    pf.load_img("11002.jpg")
    pf.show()