    
class Nonacell:
    def __init__(self, data=None, view_pos=None, view_size=None):
        if data is None:
            data = construct_data((3,3), default_value=None)
            data = [[120, 130, 140], [150, 160, 170], [180, 190, 200]]
        assert len(data) == 3
        self.data = data
        self.parent = None
        self.view_pos, self.view_size = None, None
    def __getitem__(self, index):
        result = self.data[index]
        return result
    def get_child_pos(self, subX, subY):
        return self.view_pos + self.get_child_size().real*(subX-1) + self.get_child_size().imag*(subY-1)*1j
    def get_child_size(self):
        return self.view_size/3.0
        
def rasterize_nonatree(render_target, start_corner, end_corner, data_to_draw):
    #data_to_draw = [[20, 40, 60], [80, 100, 120], [140, 160, 180]]
    assert start_corner[0] < end_corner[0]
    assert start_corner[1] < end_corner[1]
    assert isinstance(data_to_draw, Nonacell)
    
    destination_size = (end_corner[0]-start_corner[0], end_corner[1]-start_corner[1])
    subdivisionCorners = [[(start_corner[0]+int(destination_size[0]/3.0*dataX), start_corner[1]+int(destination_size[1]/3.0*dataY)) for dataX in range(4)] for dataY in range(4)]
    
    for driveDataY in range(3):
        for driveDataX in range(3):
            dataForHere = data_to_draw[driveDataY][driveDataX]
            #if driveDataY == driveDataX == 0:
            #    dataForHere = 0
            subStartCorner = subdivisionCorners[driveDataY][driveDataX]
            subEndCorner = subdivisionCorners[driveDataY+1][driveDataX+1]
            if isinstance(dataForHere, int):
                for yIndex, y in enumerate(range(subStartCorner[1], subEndCorner[1])):
                    for xIndex, x in enumerate(range(subStartCorner[0], subEndCorner[0])):
                        #if y%3 == 1 and x%3 == 1:
                        #    dataForHere = (y+x)/4
                        #if y%3 == 2 and x%3 == 2:
                        #    dataForHere = 128
                        #if yIndex == 0:
                        #    dataForHere = 128
                        #if xIndex == 0:
                        #    dataForHere = 256
                        render_target[y][x] = dataForHere
            else:
                assert isinstance(dataForHere, Nonacell)
                rasterize_nonatree(render_target, subStartCorner, subEndCorner, dataForHere)

def make_child_nonatree(parent_pos: complex, parent_size: complex, iter_limit, depth_limit, tree, subX, subY):
    newChild = make_nonatree(screen_to_complex((subX, subY), (3,3), parent_pos, parent_size, centered_sample=True), parent_size/3.0, iter_limit, depth_limit-1, center_value=tree[subY][subX])
    tree[subY][subX] = newChild
    newChild.parent = tree
                
def make_nonatree(tree_pos, tree_size, iter_limit, depth_limit, center_value=None):
    assert isinstance(center_value, int) or center_value is None
    tree = Nonacell(view_pos=tree_pos, view_size=tree_size)
    for x, y, seed in get_seeds((3,3), tree_pos, tree_size, centered_sample=True):
    #for subX, subY in range2d(3,3):
        if (center_value is not None) and x==1 and y==1:
            tree[y][x] = center_value
            continue
        #seed = tree.get_child_pos(subX, subY)
        iters = c_to_mandel_itercount_fast(seed, iter_limit, 4)
        if iters == -1:
            iters = 0
        tree[y][x] = iters
    skipCenter = False
    if sum(sum(row) for row in tree) == 0:
        skipCenter = True
    """
    if depth_limit > 1:
        for x, y, seed in get_seeds((3,3), camera_pos, view_size, centered_sample=True):
            assert isinstance(tree[y][x], int)
            if x == y == 1 and skipCenter:
                tree[y][x] = 1000
                continue
            tree[y][x] = make_nonatree(seed, view_size/3.0, iter_limit, depth_limit-1, center_value=tree[y][x])
            tree[y][x].parent = tree
    """
    if depth_limit > 1:
        for y in range(3):
            for x in range(3):
                make_child_nonatree(tree_pos, tree_size, iter_limit, depth_limit, tree, x, y)
    return tree # <------.
    
def advance_nonatree_frontier(tree, depth_limit):
    if depth_limit <= 1:
        return
    for subY, row in enumerate(tree):
        for subX, child in enumerate(row):
            if isinstance(child, Nonacell):
                advance_nonatree_frontier(child, depth_limit-1)
            else:
                assert isinstance(child, int)
                raise NotImplementedError()
                #make_child_nonatree(
        
def test_nonatree_mandelbrot(camera_pos, view_size, iter_limit, depth_limit):
    screenData = construct_data(screen.get_size(), default_value=0)
    tree = make_nonatree(camera_pos, view_size, iter_limit, depth_limit)
    rasterize_nonatree(screenData, (0,0), screen.get_size(), tree)
    for row in screenData:
        for x in range(len(row)):
            row[x] *= 10
    draw_squished_ints_to_screen([screenData])
        