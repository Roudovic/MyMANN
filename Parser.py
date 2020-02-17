import sys
import cv2
import numpy as np
import math
from Transforms import Transform

white = (255, 255, 255)
blue0 = (110, 50, 50)
blue1 = (130, 10, 255)
red = (10, 10, 255)
h = 1000
w = 1000
clear_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255


class BVHFile:
    def __init__(self, filename):
        self.filename = filename
        self.rootJoint = None
        self.currentJoint = None
        self.n_frames = 0
        self.frame_time = 0.0
        with open(filename, 'r') as f:
            assert (f.readline() == "HIERARCHY\n")
            hierarchy_part = True
            depth = 0
            line_number = 1
            start_id_motion = 0
            end_id_motion = 0
            while hierarchy_part:
                line = f.readline()
                line = line.split()
                line_number += 1
                if line[0] == "ROOT":
                    self.rootJoint = Joint(line[1])
                    self.currentJoint = self.rootJoint
                elif line[0] == "JOINT":
                    newJoint = Joint(line[1], parent=self.currentJoint)
                    self.currentJoint.children.append(newJoint)
                    self.currentJoint = newJoint
                elif line[0] == "{":
                    depth += 1
                elif line[0] == "OFFSET":
                    self.currentJoint.offset = np.array([float(line[1]), float(line[2]), float(line[3])])
                elif line[0] == "CHANNELS":
                    self.currentJoint.start_id_motion = start_id_motion
                    self.currentJoint.n_channels = int(line[1])
                    end_id_motion = start_id_motion + self.currentJoint.n_channels
                    self.currentJoint.end_id_motion = end_id_motion
                    start_id_motion = end_id_motion
                    # TODO : deal with euler angle order
                elif line[0] == "}":
                    depth -= 1
                    self.currentJoint = self.currentJoint.parent
                elif line[0] == "End":
                    newJoint = Joint("EndSite", parent=self.currentJoint)
                    self.currentJoint.children.append(newJoint)
                    self.currentJoint = newJoint

                if depth == 0 and line_number > 2:
                    hierarchy_part = False
            assert (f.readline() == "MOTION\n")
            self.n_frames = int(f.readline().split()[1])
            self.frame_time = float(f.readline().split()[2])
            line_number += 3

            lineList = np.array([np.array(line.split()) for line in f]).astype(float)
            self.rootJoint.populateMotion(lineList)

    def generateInput(self, frame):
        pass



class Joint:
    def __init__(self, name, parent=None, offset=None):
        self.name = name
        self.parent = parent
        self.offset = offset
        self.euler_order = ['z', 'x', 'y']
        self.children = []
        self.n_channels = 0
        self.start_id_motion = 0
        self.end_id_motion = 0
        self.motion_frames = None
        self.root_transform = None
        self.local_transform = None

    def populateMotion(self, lineList):
        self.motion_frames = lineList[:, self.start_id_motion: self.end_id_motion]
        for child in self.children:
            child.populateMotion(lineList)

    def computeTransform(self, frame):
        if self.parent is None:
            thetas = self.motion_frames[frame][3:6] * (2 * math.pi / 360)
            trans = self.offset + self.motion_frames[frame][0:3]
            trans =  self.motion_frames[frame][0:3]
        else:
            trans = self.offset
            thetas = self.motion_frames[frame] * (2 * math.pi / 360)
        M_local = Transform(trans, thetas).mat
        if self.parent is None:
            return M_local
        else:
            return self.parent.computeTransform(frame) @ M_local

    def draw(self, do_show=True, offset=np.array([0, 0]), canvas=np.zeros((1000, 1000, 3), np.uint8)):
        scale_skel = 5
        pt1 = self.offset[0:2] + offset
        offset_next = pt1.copy()
        cv_pt1 = tuple((scale_skel * pt1).astype(int) * np.array([1, -1]) + 500)

        cv2.putText(canvas, self.name, cv_pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue1)

        for child in self.children:
            pt2 = child.offset[0:2] + pt1
            print("Joint parent ", self.name, pt1)
            print("Joint child", child.name, pt2)

            cv_pt2 = tuple((scale_skel * pt2).astype(int) * np.array([1, -1]) + 500)

            if self.name == "Head":
                cv2.circle(canvas, cv_pt1, 30, red, -1)
            else:
                cv2.circle(canvas, cv_pt1, 10, blue1, -1)
            cv2.line(canvas, cv_pt1, cv_pt2, blue0, thickness=3)
            child.draw(False, offset_next, canvas)

        if do_show:
            cv2.imshow("Joint", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def draw_w_matrices(self, frame, center, do_show=True, canvas=None):
        if canvas is None:
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        scale_skel = 3
        origin = np.array([[0.0], [0.0], [0.0], [1.0]])
        pt1 = self.computeTransform(frame) @ origin

        cv_pt1 = (int(h / 2 + scale_skel * (pt1[0, 0] - center[0])), int(h / 2 - scale_skel * (pt1[1, 0] - center[1])))

        if self.name == "EndSite":
            cv2.circle(canvas, cv_pt1, 10, red, -1)
            cv2.putText(canvas, self.parent.name, cv_pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue1)
        if self.name == "Hips":
            cv2.circle(canvas, cv_pt1, 10, blue0, -1)
        if self.name == "Spine1":
            cv2.putText(canvas, self.name, cv_pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue1)


        for child in self.children:
            pt2 = child.computeTransform(frame) @ origin
            cv_pt2 = (int(h / 2 + scale_skel * (pt2[0, 0] - center[0])), int(h / 2 - scale_skel * (pt2[1, 0] - center[1])))
            cv2.line(canvas, cv_pt1, cv_pt2, blue0, thickness=1)
            child.draw_w_matrices(frame, center, False, canvas)

        return canvas

        # if do_show:
        #     # cv2.destroyWindow("Joint")
        #     cv2.imshow("Joint", canvas)
        #     cv2.waitKey(1)
        # return

        # cv2.destroyAllWindows()


def main():
    file1 = "MotionCapture/D1_047z_KAN01_005.bvh"
    file2 = "MotionCapture/D1_001_KAN01_001.bvh"
    file3 = "MotionCapture/D1_ex03_KAN02_006.bvh"
    bvh = BVHFile(file3)
    center = bvh.rootJoint.offset

    def play_bvh(_bvh):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output1.avi', fourcc, 20.0, (h, w))
        for frame in range(int(_bvh.n_frames)):
            img = bvh.rootJoint.draw_w_matrices(frame, center=center)
            out.write(img)
            img = clear_canvas.copy()
        out.release()
    play_bvh(bvh)
    return 0




if __name__ == "__main__":
    sys.exit(main())
