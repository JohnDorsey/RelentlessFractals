
class View:
    def __init__(self, center_pos, size):
        assert isinstance(center_pos, complex)
        assert isinstance(size, complex)
        self.center_pos, self.size = (center_pos, size)
        self.corner_pos = self.center_pos - (self.size / 2.0)
        
    def get_sub_view_size(self, subdivisions_pair):
        return parallel_div_complex_by_floats(self.size, subdivisions_pair)
    
    """
    def get_sub_view_corner(self, subdivisions_pair, sub_view_coord):
        # assert all(sub_view_coord[i] <= subdivisions_pair[i] for i in (0,1))
        return self.corner_pos + parallel_mul_complex_by_floats(self.get_sub_view_size(subdivisions_pair), sub_view_coord)

    def get_sub_view_center(self, subdivisions_pair, sub_view_coord):
        return self.get_sub_view_corner(subdivisions_pair, (sub_view_coord[0] + 0.5, sub_view_coord[1] + 0.5))
    """
    





class GridSettings:
    
    def __init__(self, view, grid_size):
        assert all(is_round_binary(item) for item in grid_size)
        self.grid_size = tuple(iter(grid_size)) # make it a tuple as a standard for equality tests in other places.
        self.view = view
        assert self.view.size.real > 0.0
        assert self.view.size.imag > 0.0
        
        self.cell_size = view.get_sub_view_size(self.grid_size)
        self.graveyard_point = (256.5 + abs(self.view.center_pos) + abs(2.0*self.view.size)) # a complex coordinate that will never appear on camera. Make it so large that there is no doubt.
        
    def whole_to_complex(self, whole_coord, centered=None):
        assert centered is not None
        return self.view.corner_pos + parallel_mul_complex_by_floats(self.cell_size, (bump(whole_coord, 0.5) if centered else whole_coord))
        
    def complex_to_whole(self, complex_coord, centered=None):
        # centered_sample might not be logically needed for the answer to this question, depending on how the screen is defined in future versions of the program.
        complexInView = complex_coord - self.view.corner_pos
        complexOfCell = parallel_div_complex_by_complex(complexInView, self.cell_size)
        return (int(complexOfCell.real), int(complexOfCell.imag))
        
    def complex_to_item(self, data, complex_coord, centered=None):
        assert len(data) == self.grid_size[1] and len(data[0]) == self.grid_size[0]
        try:
            x, y = self.complex_to_whole(complex_coord, centered=centered)
        except OverflowError:
            raise CoordinateError("near-infinity can never be a list index!")
        if x < 0 or y < 0:
            raise CoordinateError("negatives not allowed here.")
        if y >= len(data):
            raise CoordinateError("y is too high.")
        row = data[y]
        if x >= len(row):
            raise CoordinateError("x is too high.")
        return row[x]
        
    def iter_cell_whole_coords(self, range_descriptions=None, swap_iteration_order=False):
        if range_descriptions is None:
            range_descriptions = [(0, s, 1) for s in self.grid_size]
        iterationOrder = [1,0]
        if swap_iteration_order:
            iterationOrder = iterationOrder[::-1]
        return higher_range(range_descriptions, iteration_order=iterationOrder)
        
    def iter_cell_descriptions(self, range_descriptions=None, swap_iteration_order=False, centered=None):
        assert centered is not None
        for x, y in self.iter_cell_whole_coords(range_descriptions=range_descriptions, swap_iteration_order=swap_iteration_order):
            yield (x, y, self.whole_to_complex((x,y), centered=centered))
            
testGrid = GridSettings(View(0+0j, 4+4j), (2,2))
assert_equal(list(testGrid.iter_cell_whole_coords()), [(0, 0), (1, 0), (0, 1), (1, 1)])
SegmentGeometry.assert_nearly_equal(list(testGrid.iter_cell_descriptions(centered=False)), [(0, 0, -2-2j), (1, 0, 0-2j), (0, 1, -2+0j), (1, 1, 0+0j)])
SegmentGeometry.assert_nearly_equal(list(testGrid.iter_cell_descriptions(centered=True)), [(0, 0, -1-1j), (1, 0, 1-1j), (0, 1, -1+1j), (1, 1, 1+1j)])
del testGrid