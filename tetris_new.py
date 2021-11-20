import random
import cv2
import numpy as np
from PIL import Image
from time import sleep

# TetrisNew game class


class TetrisNew:

    '''TetrisNew game class'''

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: {  # I
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            90: [(1, 0), (1, 1), (1, 2), (1, 3)],
            180: [(3, 0), (2, 0), (1, 0), (0, 0)],
            270: [(1, 3), (1, 2), (1, 1), (1, 0)],
        },
        1: {  # T
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 1), (1, 2), (1, 1), (1, 0)],
            180: [(1, 2), (2, 1), (1, 1), (0, 1)],
            270: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # L
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 0)],
            180: [(1, 2), (1, 1), (1, 0), (0, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # J
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 2)],
            180: [(1, 2), (1, 1), (1, 0), (2, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # Z
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(0, 2), (0, 1), (1, 1), (1, 0)],
            180: [(2, 1), (1, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (0, 1), (0, 2)],
        },
        5: {  # S
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(0, 1), (1, 1), (1, 0), (2, 0)],
            270: [(1, 2), (1, 1), (0, 1), (0, 0)],
        },
        6: {  # O
            0: [(1, 0), (2, 0), (1, 1), (2, 1)],
            90: [(1, 0), (2, 0), (1, 1), (2, 1)],
            180: [(1, 0), (2, 0), (1, 1), (2, 1)],
            270: [(1, 0), (2, 0), (1, 1), (2, 1)],
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }

    def __init__(self):
        self.reset()

    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[0] * TetrisNew.BOARD_WIDTH for _ in range(TetrisNew.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(TetrisNew.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        self.col_idx = np.concatenate([np.arange(TetrisNew.BOARD_HEIGHT)]
                                      * TetrisNew.BOARD_WIDTH).reshape(-1, TetrisNew.BOARD_HEIGHT).T

        return self._get_board_props(self.board)

    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return TetrisNew.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = TetrisNew.MAP_PLAYER
        return board

    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score

    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(TetrisNew.TETROMINOS)))
            random.shuffle(self.bag)

        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= TetrisNew.BOARD_WIDTH \
                    or y < 0 or y >= TetrisNew.BOARD_HEIGHT \
                    or self.board[y][x] == TetrisNew.MAP_BLOCK:
                return True
        return False

    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = TetrisNew.MAP_BLOCK
        return board

    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(
            board) if sum(row) == TetrisNew.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(TetrisNew.BOARD_WIDTH)])
        return len(lines_to_clear), board

    def _highest_idx(self, board, block_val, equiv=True):
        np_board = np.array(board)
        idx = (np_board == block_val) if equiv else (np_board != block_val)
        highest_idx = np.argmax(idx, axis=0)
        highest_idx = np.where(idx.any(axis=0), highest_idx, TetrisNew.BOARD_HEIGHT)

        return highest_idx

    def _number_of_holes(self, board):
        '''Number of holes in the board (empty square with at least one block above it)'''

        holes = 0
        highest_block_idx = self._highest_idx(board, block_val=TetrisNew.MAP_BLOCK, equiv=True)
        np_board = np.array(board)

        return np.sum((np_board == TetrisNew.MAP_EMPTY) & (self.col_idx > highest_block_idx))

    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0

        highest_block_idx = self._highest_idx(board, block_val=TetrisNew.MAP_BLOCK)
        bumpiness = np.abs(np.diff(highest_block_idx))

        return np.sum(bumpiness), np.max(bumpiness)

    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = TetrisNew.BOARD_HEIGHT

        highest_nonempty_idx = self._highest_idx(board, block_val=TetrisNew.MAP_EMPTY, equiv=False)
        height = TetrisNew.BOARD_HEIGHT - highest_nonempty_idx

        return np.sum(height), np.max(height), np.min(height)

    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece

        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = TetrisNew.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, TetrisNew.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states

    def get_state_size(self):
        '''Size of the state'''
        return 4

    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        score = 1 + (lines_cleared ** 2) * TetrisNew.BOARD_WIDTH
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over

    def render(self):
        '''Renders the current board'''
        img = [TetrisNew.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(
            TetrisNew.BOARD_HEIGHT, TetrisNew.BOARD_WIDTH, 3).astype(
            np.uint8)
        img = img[..., ::-1]  # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((TetrisNew.BOARD_WIDTH * 25, TetrisNew.BOARD_HEIGHT * 25), Image.NEAREST)
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)
