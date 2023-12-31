import pygame as py
import sys
import copy

py.font.init()

arial = "Arial"

END_FONT = py.font.SysFont(arial, 70, bold=True)

FPS = 60

WIDTH = 650
HEIGHT = 650

WIN = py.display.set_mode((WIDTH, HEIGHT))

# -------------------------------- CALCING BLOCK LENGTHS ------------------------ #
border_length = 21
NUM_GRID = 8  # 8x8 grid
SQUARE_SIZE = (WIDTH - 2*border_length) / NUM_GRID


GRIDS = []
coords = []

off_y = HEIGHT - SQUARE_SIZE - border_length
for y in range(NUM_GRID):
    off_x = border_length
    for x in range(NUM_GRID):
        coords.append((x*SQUARE_SIZE + border_length, HEIGHT -
                      (y+1)*SQUARE_SIZE - border_length))
        GRIDS.append(py.Rect(off_x, off_y, SQUARE_SIZE, SQUARE_SIZE))
        off_x += (SQUARE_SIZE)

    off_y -= (SQUARE_SIZE)

# test to see specific grid
grid_idx = 0
TEST_SURF = py.Surface((GRIDS[grid_idx].width, GRIDS[grid_idx].height))
TEST_SURF.fill((255, 0, 0))
# ---------------------------------------------------------------------- #


peices = ["pw",
          "pb",
          "kw",
          "kb",
          "qw",
          "qb",
          "bw",
          "bb",
          "nw",
          "nb",
          "rb",
          "rw"]

# -------------------------------- THEMES ------------------------ #

# region PIECES THEME EXPLANATION
# 0 -> my screenshotted ones from chess.com
# 1 -> Cute
# 2 -> Cute
# 3 -> plain
# 4 -> Cute
# 5 -> Magnetic
# 6 -> Shadow
# 7 -> Standard
# 8 -> Fancy
# 9 -> Cute
# 10 -> Aztec
# 11 -> Blobby Cute
# 12 -> Punk Rock (cool)
# 13 -> Cutest?
# 14 -> Cute
# 15 -> Minimilistic
# 16 -> Symmetrical (off-putting)
# 17 -> Letters
# endregion

# region BOARD THEME EXPLANATION
# 0 -> wood (light)
# 1 -> wood (basic)
# 2 -> White scratch
# 3 -> Asphalt (dirty version)
# 4 -> Purple (clean)
# 5 -> Orange, White, Blue
# 6 -> Dark Blue, Orange
# 7 -> Dark Purple, orange (i guess?)
# 8 -> Green (clean)
# 9 -> Rainbow
# 10 -> Asphalt (clean version)
# endregion

PIECE_THEME = 12
BOARD_THEME = 1


P_W = 70
P_H = P_W

PIECE_DICT = {elem: py.transform.scale(py.image.load(f"Assets/Pieces/theme{PIECE_THEME}/{elem}.png"), (P_W, P_H))
              for elem in peices}

BIG_PIECE_DICT = {elem: py.transform.scale(py.image.load(f"Assets/Pieces/theme{PIECE_THEME}/{elem}.png"), (P_W+10, P_H+10))
                  for elem in peices}

BOARD = py.transform.scale(py.image.load(
    f"Assets/boards/theme{BOARD_THEME}.png"), (WIDTH, HEIGHT))

# hgihlight colours

RED = (255, 0, 0, 100)
GREEN = (0, 120, 0, 100)

YELLOW = (245, 163, 49, 150)
GREY = (0, 0, 0, 75)

HIGHLIGHT_MOVES_COL = RED
HIGHLIGHT_PREV_COL = GREY

# ---------------------------------------------------------------------- #


# ------------------------- VARIABLES USED IN PROGRAM ------------- #
# bitboard
BB = {
    "white": {
        "p": 0xFF00,
        "k": 0x10,
        "r": 0x81,
        "n": 0x42,
        "b": 0x24,
        "q": 0x08
    },
    "black": {
        "p": 0xFF00 << (8*5),
        "k": 0x10 << (8*7),
        "r": 0x81 << (8*7),
        "n": 0x42 << (8*7),
        "b": 0x24 << (8*7),
        "q": 0x08 << (8*7)
    }

}

BLACK_BB = 0xFFFF000000000000
WHITE_BB = 0x000000000000FFFF

# enPesant
DOUBLE_PAWN_FILE = None

# Castling
# [white queen side, white king side, black queen side, black king side]
CASTLING_MASKS = [0, 0, 0, 0]

# if rights are revoked then it gets replaced by this thing
CASTLING_MASKS_REVOKED = [0x000000000000000E,
                          0x0000000000000060,
                          0x0E00000000000000,
                          0x6000000000000000]


def bin_2_sqr(bb):

    for n in range(64):
        if (bb >> n) & 1:
            return n
    return None


def reset_bb():

    for (col_key, col_val) in BB.items():
        for (key, val) in col_val.items():
            BB[col_key][key] = 0

    set_entire_bb()


def fen_2_bb(fen):
    global BB, DOUBLE_PAWN_FILE

    reset_bb()

    # split into each peice of info ([structure, turn, castling rights, enpesant, last 2 idk])
    fen_attr = fen.split(" ")

    # ----------------------------- structure ----------------------------- #
    fen_struct = fen_attr[0].split("/")
    fen_struct.reverse()

    idx = 0
    for elem in fen_struct:
        for ltr in elem:
            try:
                idx += int(ltr)

            except:
                if ltr.isupper():
                    col = "white"
                else:
                    col = "black"

                lower_ltr = ltr.lower()

                BB[col][lower_ltr] |= 0b1 << idx

                idx += 1

    set_entire_bb()

    # ----------------------------- Turns ----------------------------- #
    fen_turn = fen_attr[1]
    # if white -> 0, if black -> 1 (works nicely with my system
    turn = (fen_turn == "b")

    # ----------------------------- Castling ----------------------------- #
    castling_specification = fen_attr[2]

    # recall, [white queen side, white king side, black queen side, black king side]
    for i in range(4):
        # first revoke all then restore if FEN says so
        CASTLING_MASKS[i] = CASTLING_MASKS_REVOKED[i]
    for elem in castling_specification:
        if elem == "Q":
            CASTLING_MASKS[0] = 0
        elif elem == "K":
            CASTLING_MASKS[1] = 0
        elif elem == "q":
            CASTLING_MASKS[2] = 0
        elif elem == "k":
            CASTLING_MASKS[3] = 0

    # ----------------------------- En Pesant ----------------------------- #
    # the way our system works, all we need is the col
    enPesant_spec = fen_attr[3]

    if enPesant_spec == "-":
        pass
    else:
        col = ord(enPesant_spec[0]) - 97
        DOUBLE_PAWN_FILE = col

    return turn


def give_piece(idx):

    sqr_bb = 1 << idx
    for (col_key, col_val) in BB.items():
        for (key, val) in col_val.items():
            if val & sqr_bb:
                return col_key, key

    return 0, 0


def pickup(idx, p_col, p_type, mouse_pos, legal_sqrs):

    update_screen()

    disp_sqrs(legal_sqrs, idx)

    WIN.blit(BIG_PIECE_DICT[f"{p_type}{p_col[0]}"],
             (mouse_pos[0] - P_W//2, mouse_pos[1] - P_W//2))


def update_screen():

    WIN.blit(BOARD, (0, 0))

    offset = (SQUARE_SIZE - P_W) / 2
    for (col_key, col_val) in BB.items():
        for (key, val) in col_val.items():
            for i in range(0, 64):
                if (val >> i & 1):
                    WIN.blit(PIECE_DICT[f"{key}{col_key[0]}"],
                             (coords[i][0] + offset, coords[i][1] + offset))


def update_bb(p_col, p_type, prev_sqr, curr_sqr):
    global DOUBLE_PAWN_FILE, BB
    old_BB = copy.deepcopy(BB)

    # ---------------- get sqr_bb -------------- #
    prev_sqr_bb = 1 << prev_sqr
    curr_sqr_bb = 1 << curr_sqr
    # ------------------------------------------ #

    cur_col = curr_sqr % 8
    prev_col = prev_sqr % 8

    if p_col == "white":
        opp_col = "black"
    else:
        opp_col = "white"

    # ---------------- castling -------------- #

    if p_type == "k":

        col_offset = cur_col - prev_col

        if abs(col_offset) > 1:

            # cant castle though or out of check
            # which sqaure to check for in check?

            # we have curr row, we have sqaures
            # could use p_col
            if p_col == "white":
                row_in_question = 0
            else:
                row_in_question = 7

            # might aswell specify which rook we want to swap if we not in check
            if (col_offset) < 0:
                # queen side
                cols_in_question = (2, 3, 4)
                old_rook_col = 0
                new_rook_col = 3

            else:
                # king side
                cols_in_question = (4, 5, 6)
                old_rook_col = 7
                new_rook_col = 5

            for elem in cols_in_question:
                check_sqr = row_in_question*8 + elem
                if check_checker(p_col, sqr=check_sqr):
                    return 0

            # ight now swap the rook and things
            BB[p_col]["k"] ^= (prev_sqr_bb | curr_sqr_bb)
            old_rook_bb = 1 << (row_in_question*8) + old_rook_col
            new_rook_bb = 1 << (row_in_question*8) + new_rook_col
            BB[p_col]["r"] ^= (old_rook_bb | new_rook_bb)

            # now rovoke that cols right to castle
            if row_in_question == 8:
                indices = (2, 3)
            else:
                indices = (0, 1)
            for elem in indices:
                CASTLING_MASKS[elem] = CASTLING_MASKS_REVOKED[elem]
            return 1

    # ---------------- captures ----------------- #
    # normal
    captured = None

    for (key, val) in BB[opp_col].items():
        if val & curr_sqr_bb:
            captured = (opp_col, key)
            break

    # reset captured piece
    if not captured is None:
        BB[captured[0]][captured[1]] ^= curr_sqr_bb

    # enpesant
    if p_type == "p" and captured is None and prev_sqr % 8 != curr_sqr % 8:
        if p_col == "white":
            remove_sqr = curr_sqr - 8
        else:
            remove_sqr = curr_sqr + 8

        BB[opp_col]["p"] ^= 1 << remove_sqr

    # ------------------------------------------ #

    # -------------------- changes made to bb ------------------ #

    BB[p_col][p_type] ^= (prev_sqr_bb | curr_sqr_bb)
    set_entire_bb()
    # ------------------------------------------ #

    # -------------------- DID MOVE JUST MADE CAUSE CHECK ------------------ #
    if check_checker(opp_col):

        if checkmate_checker(opp_col):
            # print("checkmate")
            return "checkmate"
        else:
            # print("CHECKED")
            pass
    # ------------------------------------------ #

    # -------------------- cant make move that will cause check ------------------ #
    if check_checker(p_col):
        BB = old_BB
        set_entire_bb()
        return 0
    # ------------------------------------------ #

    # --------------------- handeling en pesant -------------- #
    if p_type == "p":
        pr = prev_sqr // 8
        cr = curr_sqr // 8
        if abs(pr - cr) == 2:
            DOUBLE_PAWN_FILE = curr_sqr % 8
            return 1
        if ((p_col == "white") and (curr_sqr // 8 == 7)) or ((p_col == "black") and (curr_sqr // 8 == 0)):
            choose_promotion(p_col, curr_sqr_bb)

    DOUBLE_PAWN_FILE = None

    # ------------------------------------------ #

    # [white queen side, white king side, black queen side, black king side]
    # --------------------- check_castle -------------- #
    if p_type == "k":
        if p_col == "white":
            rights_to_revoke = (0, 1)
        else:
            rights_to_revoke = (2, 3)

        for elem in rights_to_revoke:
            CASTLING_MASKS[elem] = CASTLING_MASKS_REVOKED[elem]

    if p_type == "r":

        if p_col == "white":
            if prev_col == 7:
                CASTLING_MASKS[1] = CASTLING_MASKS_REVOKED[1]
            else:
                CASTLING_MASKS[0] = CASTLING_MASKS_REVOKED[0]
        else:
            if prev_col == 0:
                CASTLING_MASKS[2] = CASTLING_MASKS_REVOKED[2]
            else:
                CASTLING_MASKS[3] = CASTLING_MASKS_REVOKED[3]

    if not captured is None and captured[1] == "r":

        if p_col == "black":
            if cur_col == 7:
                CASTLING_MASKS[1] = CASTLING_MASKS_REVOKED[1]
            else:
                CASTLING_MASKS[0] = CASTLING_MASKS_REVOKED[0]
        else:
            if cur_col == 0:
                CASTLING_MASKS[2] = CASTLING_MASKS_REVOKED[2]
            else:
                CASTLING_MASKS[3] = CASTLING_MASKS_REVOKED[3]

    # ------------------------------------------ #

    return 1


def get_indices(legal_moves):

    indices = []
    for n in range(64):
        if (legal_moves >> n & 1):
            indices.append(n)

    return indices


def movement_mask(p_type, sqr):

    legal_moves = 0

    if p_type == "n":
        directions = [(2, -1), (2, 1), (-2, -1), (-2, 1),
                      (1, -2), (-1, -2), (1, 2), (-1, 2)]

        for dr, dc in directions:
            row = sqr // 8 + dr
            col = sqr % 8 + dc

            if 0 <= row <= 7 and 0 <= col <= 7:

                target_bb = 1 << (col + 8*row)
                legal_moves |= target_bb

        return legal_moves

    if p_type == "k":
        directions = [(1, -1), (1, 0), (1, 1),
                      (0, -1), (0, 1),
                      (-1, -1), (-1, 0), (-1, 1)]

        for dr, dc in directions:

            row = sqr // 8 + dr
            col = sqr % 8 + dc

            if 0 <= row <= 7 and 0 <= col <= 7:

                target_bb = 1 << (col + 8*row)
                legal_moves |= target_bb

        return legal_moves

    if p_type == "r":

        directions = [(1, 0),
                      (0, 1),
                      (-1, 0),
                      (0, -1)]

    if p_type == "b":
        directions = [(1, 1),
                      (-1, 1),
                      (1, -1),
                      (-1, -1)]

    for dr, dc in directions:

        row = sqr // 8 + dr
        col = sqr % 8 + dc

        mincol = 1
        maxcol = 6
        minrow = 1
        maxrow = 6

        if sqr // 8 == 0:
            minrow = 0
        if sqr // 8 == 7:
            maxrow = 7

        if sqr % 8 == 0:
            mincol = 0
        if sqr % 8 == 7:
            maxcol = 7

        while minrow <= row <= maxrow and mincol <= col <= maxcol:

            target_bb = 1 << (col + 8*row)
            target_sqr = col + 8*row

            target_bb = (1 << target_sqr)

            legal_moves |= target_bb

            row += dr
            col += dc

    return legal_moves


def arrangments(p_type, sqr):

    move_mask = movement_mask(p_type, sqr)

    index_1s = []
    for n in range(64):
        if ((move_mask >> n) & 1) == 1:
            index_1s.append(n)

    num1s = len(index_1s)

    poss_arrs = pow(2, num1s)

    all_bbs = []
    all_legals = []

    for arr_num in range(poss_arrs):
        block_mask = 0

        # ------- implentation1 -------- #

        # for msb_arr in range(num1s):
        #     if arr_num >> msb_arr == 1:
        #         break

        # for shift in range(msb_arr + 1):
        #     block_mask |= (((arr_num >> shift) & 1) << index_1s[shift])

        # all_bbs.append(block_mask)

    # -------- implement 2 ---------- #

        for shift in range(num1s):
            block_mask |= (((arr_num >> shift) & 1) << index_1s[shift])

    # ------------------------------ #

    # ---------------- get legal moves ------------ #

        if p_type == "r":
            directions = [(1, 0),
                          (0, 1),
                          (-1, 0),
                          (0, -1)]

        if p_type == "b":
            directions = [(1, 1),
                          (-1, 1),
                          (1, -1),
                          (-1, -1)]

        legal_moves = 0

        for dr, dc in directions:

            row = sqr // 8 + dr
            col = sqr % 8 + dc

            mincol = 1
            maxcol = 6
            minrow = 1
            maxrow = 6

            if sqr // 8 == 0:
                minrow = 0
            if sqr // 8 == 7:
                maxrow = 7

            if sqr % 8 == 0:
                mincol = 0
            if sqr % 8 == 7:
                maxcol = 7

            while minrow <= row <= maxrow and mincol <= col <= maxcol:
                target_bb = 1 << (col + 8*row)

                legal_moves |= target_bb

                if target_bb & block_mask:
                    break

                row += dr
                col += dc

    # ------------------------------ #

        all_bbs.append(block_mask)
        all_legals.append(legal_moves)

    return (all_bbs, all_legals)


def disp_sqrs(sqrs: list, sqr_selected):
    surf = py.Surface((SQUARE_SIZE, SQUARE_SIZE), py.SRCALPHA)
    surf.fill(HIGHLIGHT_MOVES_COL)

    sel_surf = py.Surface((SQUARE_SIZE, SQUARE_SIZE), py.SRCALPHA)
    sel_surf.fill(HIGHLIGHT_PREV_COL)

    for elem in sqrs:
        WIN.blit(surf, coords[elem])

    WIN.blit(sel_surf, coords[sqr_selected])


def set_entire_bb():
    global BLACK_BB, WHITE_BB

    BLACK_BB = (BB["black"]["p"] |
                BB["black"]["r"] |
                BB["black"]["b"] |
                BB["black"]["n"] |
                BB["black"]["q"] |
                BB["black"]["k"])

    WHITE_BB = (BB["white"]["p"] |
                BB["white"]["r"] |
                BB["white"]["b"] |
                BB["white"]["n"] |
                BB["white"]["q"] |
                BB["white"]["k"])


def give_legal_moves(p_col, p_type, sqr):

    row = sqr // 8
    col = sqr % 8

    entire_bb = (WHITE_BB | BLACK_BB) ^ 1 << sqr

    if p_col == "black":
        friend_bb = BLACK_BB
        opp_bb = WHITE_BB
    else:
        friend_bb = WHITE_BB
        opp_bb = BLACK_BB

    if p_type == "r":

        sqaures_to_check = (1, 6)
        sqaures_to_add = (0, 7)

        move_mask = movement_mask("r", sqr)
        my_idx = (sqr, entire_bb & move_mask)
        legal_bb = ROOK_LKUP_TBL[my_idx]

        # add edges for rook

        legal_bb ^= 1 << sqr

        for idx in range(2):
            # along row
            sqr_to_check = 1 << (row*8 + sqaures_to_check[idx])
            if (sqr_to_check & legal_bb) and not (sqr_to_check & entire_bb):
                legal_bb |= 1 << ((row*8) + sqaures_to_add[idx])

            # along col
            sqr_to_check = 1 << (sqaures_to_check[idx]*8 + col)
            if (sqr_to_check & legal_bb) and not (sqr_to_check & entire_bb):
                legal_bb |= 1 << ((sqaures_to_add[idx]*8) + col)

        legal_bb ^= 1 << sqr

    elif p_type == "b":

        col_1 = 0x0202020202020202
        col_6 = 0x4040404040404040
        row_1 = 0x000000000000FF00
        row_6 = 0x00FF000000000000

        collision_mask = [col_1, col_6, row_1, row_6]

        move_mask = movement_mask("b", sqr)
        my_idx = (sqr, entire_bb & move_mask)
        legal_bb = BISH_LKUP_TBL[my_idx]

        sqrs_done = []
        bb_to_add = 0
        # legal_bb ^= 1 << sqr

        if col == 6:
            for i in range(-1, 2, 2):
                new_row = row+i
                if 0 <= new_row <= 7:
                    bb_to_add |= 1 << (new_row*8 + col+1)

            collision_mask[1] = None

        elif col == 1:
            for i in range(-1, 2, 2):
                new_row = row+i
                if 0 <= new_row <= 7:
                    bb_to_add |= 1 << (new_row*8 + col-1)
            collision_mask[0] = None

        if row == 6:
            for i in range(-1, 2, 2):
                new_col = col+i
                if 0 <= new_col <= 7:
                    bb_to_add |= 1 << ((row+1)*8 + new_col)

            collision_mask[3] = None

        elif row == 1:
            for i in range(-1, 2, 2):
                new_col = col+i
                if 0 <= new_col <= 7:
                    bb_to_add |= 1 << ((row-1)*8 + new_col)

            collision_mask[2] = None

        for elem in collision_mask:
            if elem is None:
                continue
            collision = legal_bb & elem
            while collision:
                pot_sqr = bin_2_sqr(collision)
                pot_bb = 1 << pot_sqr

                if not (pot_sqr in sqrs_done) and (not (pot_bb & entire_bb)):

                    pot_row = pot_sqr // 8
                    pot_col = pot_sqr % 8

                    if pot_row > row:
                        pot_row += 1
                    elif pot_row < row:
                        pot_row -= 1

                    if pot_col > col:
                        pot_col += 1
                    elif pot_col < col:
                        pot_col -= 1

                    bb_to_add |= (1 << ((pot_row*8) + pot_col))
                    sqrs_done.append(pot_sqr)

                collision -= pot_bb

        legal_bb |= bb_to_add
        # legal_bb ^= 1 << sqr

    elif p_type == "q":

        rook_legal_bb = give_legal_moves(p_col, "r", sqr)

        move_mask = movement_mask("b", sqr)
        my_idx = (sqr, entire_bb & move_mask)
        bishop_legal_bb = give_legal_moves(p_col, "b", sqr)

        legal_bb = rook_legal_bb | bishop_legal_bb

    elif p_type == "k":

        legal_bb = movement_mask("k", sqr)

        # decides ehich row
        if p_col == "white":
            castling_checking = CASTLING_MASKS[:2]
            checking_sqaures = CASTLING_MASKS_REVOKED[:2]

        else:
            castling_checking = CASTLING_MASKS[2:]
            checking_sqaures = CASTLING_MASKS_REVOKED[2:]

        # decides which col
        # [QUEEN SIDE, KING SIDE]
        target_sqrs = entire_bb & checking_sqaures[0]
        if not (castling_checking[0] | target_sqrs):
            legal_bb |= (1 << (row*8 + col-2))

        target_sqrs = entire_bb & checking_sqaures[1]

        if not (castling_checking[1] | target_sqrs):
            legal_bb |= 1 << (row*8 + col+2)

    elif p_type == "n":
        legal_bb = movement_mask("n", sqr)

    elif p_type == "p":

        legal_bb = 0

        # ---------- check for up or down and if double pawn move or not ------------ #
        if p_col == "white":
            directions = [(1, 0)]
            if (row == 1):
                directions += [(2, 0)]
        else:
            directions = [(-1, 0)]
            if (row == 6):
                directions += [(-2, 0)]

        # -------------------------------------------------------------------------- #

        # ---------- check for take in front left and right ------------ #
        for i in range(-1, 2, 2):
            check_col = col + i
            if 0 <= check_col <= 7:
                check_sqr = (row + directions[0][0])*8 + (check_col)

                if (1 << check_sqr) & opp_bb:
                    legal_bb |= (1 << check_sqr)
        # -------------------------------------------------------------------------- #

        # ---------- enPesant checking  ------------ #
        # (if enpesant exists) and (pawn is on either side) and (we are on correct row then legal)
        if not DOUBLE_PAWN_FILE is None:
            if (abs(col - DOUBLE_PAWN_FILE) == 1):
                if (p_col == "white" and row == 4) or (p_col == "black" and row == 3):
                    enPesant_sqr = (
                        row + directions[0][0])*8 + DOUBLE_PAWN_FILE
                    legal_bb |= 1 << enPesant_sqr
        # -------------------------------------------------------------------------- #

        # ---------- then just the normal shit  ------------ #
        # using looping cause dont wanna do lookup table

        for dr, dc in directions:
            running_col = col + dc
            running_row = row + dr

            if 0 <= running_row <= 7 and 0 <= running_col <= 7:

                target_sqr = running_col + 8*running_row

                # ---------- blockers ---------- #
                target_bb = 1 << target_sqr
                if target_bb & (BLACK_BB | WHITE_BB):
                    break
                # ------------------------------ #

                legal_bb |= target_bb
            # -------------------------------------------------------------------------- #

        return legal_bb

    legal_bb ^= (legal_bb & friend_bb)

    return legal_bb


def checkmate_checker(checked_col):
    global BB
    # starting to think cycling through every move is the best idea
    # but again first check if king aleviates check then queen, rook, bishop, knight, pawn

    old_bb = copy.deepcopy(BB)
    king_sqr = bin_2_sqr(BB[checked_col]["k"])

    # --------------------------- king ------------------------ #
    # does a king move aleviate check?

    king_moves = give_legal_moves(checked_col, "k", king_sqr)
    BB[checked_col]["k"] ^= 1 << king_sqr
    set_entire_bb()

    while king_moves:
        check_sqr = bin_2_sqr(king_moves)
        if not check_checker(checked_col, sqr=check_sqr):
            BB[checked_col]["k"] ^= 1 << king_sqr
            set_entire_bb()
            return 0
        king_moves -= 1 << check_sqr

    BB[checked_col]["k"] ^= 1 << king_sqr
    set_entire_bb()

    # --------------------------- rest of peices ------------------------ #
    pieces = ["q", "r", "b", "n", "p"]

    for elem in pieces:
        peice_bb = BB[checked_col][elem]
        while peice_bb:
            peice_sqr = bin_2_sqr(peice_bb)
            peice_moves_bb = give_legal_moves(checked_col, elem, peice_sqr)

            while peice_moves_bb:

                # BB[checked_col][elem] &= ~(1 << peice_sqr)
                check_sqr = bin_2_sqr(peice_moves_bb)

                update_bb(checked_col, elem, peice_sqr, check_sqr)

                if not check_checker(checked_col, king_sqr):
                    BB = copy.deepcopy(old_bb)
                    set_entire_bb()
                    return 0

                BB = copy.deepcopy(old_bb)
                set_entire_bb()
                peice_moves_bb -= 1 << check_sqr
            peice_bb -= 1 << peice_sqr
    return 1


def check_checker(col_2_check, sqr=None, give_mask=False):

    if col_2_check == "white":
        opp_col = "black"
    else:
        opp_col = "white"

    if sqr is None:
        sqr = bin_2_sqr(BB[col_2_check]["k"])

    if not give_mask:
        # bishop_check?
        bish_moves = give_legal_moves(col_2_check, "b", sqr)
        if bish_moves & BB[opp_col]["b"]:
            return 1

        # rook_check?
        rook_moves = give_legal_moves(col_2_check, "r", sqr)
        if rook_moves & BB[opp_col]["r"]:
            return 1

        if (rook_moves | bish_moves) & BB[opp_col]["q"]:
            return 1

        if give_legal_moves(col_2_check, "n", sqr) & BB[opp_col]["n"]:
            return 1

        if (give_legal_moves(col_2_check, "p", sqr) & BB[opp_col]["p"]):
            return 1

        return 0
    else:
        check_mask = 0
        # bishop_check?
        bish_moves = give_legal_moves(col_2_check, "b", sqr)
        if bish_moves & BB[opp_col]["b"]:
            check_mask |= bish_moves

        # rook_check?
        rook_moves = give_legal_moves(col_2_check, "r", sqr)
        if rook_moves & BB[opp_col]["r"]:
            check_mask |= rook_moves

        if (rook_moves) & BB[opp_col]["q"]:
            check_mask |= rook_moves

        if (bish_moves) & BB[opp_col]["q"]:
            check_mask |= bish_moves

        if give_legal_moves(col_2_check, "n", sqr) & BB[opp_col]["n"]:
            check_mask |= BB[opp_col]["n"]

        if (give_legal_moves(col_2_check, "p", sqr) & BB[opp_col]["p"]):
            check_mask |= BB[opp_col]["p"]

        return check_mask


def choose_promotion(col, targeted_bb):

    BB[col]["p"] ^= targeted_bb
    clock = py.time.Clock()
    run = 1

    # can just play around with these
    box_size = 100
    spacing = 20
    first_box_y = (HEIGHT / 2) - box_size / 2
    first_box_x = WIDTH / 2 - (4*box_size + 3*spacing)/2

    white = (255, 255, 255, 150)
    black = (0, 0, 0, 150)
    red = (150, 0, 0, 150)

    orange = (245, 190, 49, 200)
    fg_col = white
    highlight_col = (245, 190, 49, 200)

    floating_box = py.Surface((box_size, box_size), py.SRCALPHA)
    highlight_index = 0

    piece_disp_dict = {
        0: "q",
        1: "r",
        2: "b",
        3: "n"
    }

    BB[col]["q"] ^= targeted_bb

    while run:
        update_screen()
        for i in range(4):
            if i == highlight_index:
                floating_box.fill(highlight_col)
                piece_2_disp = BIG_PIECE_DICT[f"{piece_disp_dict[i]}{col[0]}"]
                offset = (box_size - P_W-10) / 2

            else:
                piece_2_disp = PIECE_DICT[f"{piece_disp_dict[i]}{col[0]}"]
                floating_box.fill(fg_col)
                offset = (box_size - P_W) / 2

            x = first_box_x + i*(box_size+spacing)
            WIN.blit(floating_box, (x, first_box_y))

            WIN.blit(piece_2_disp,
                     (x+offset, first_box_y + offset))

        clock.tick(FPS)
        py.display.update()

        for event in py.event.get():
            if event.type == py.KEYDOWN:
                if event.key == py.K_RETURN:
                    run = 0
                elif event.key == py.K_LEFT:
                    BB[col][piece_disp_dict[highlight_index]] ^= targeted_bb
                    highlight_index -= 1
                    if highlight_index < 0:
                        highlight_index = 3
                    BB[col][piece_disp_dict[highlight_index]] ^= targeted_bb

                elif event.key == py.K_RIGHT:
                    BB[col][piece_disp_dict[highlight_index]] ^= targeted_bb
                    highlight_index += 1
                    if highlight_index > 3:
                        highlight_index = 0
                    BB[col][piece_disp_dict[highlight_index]] ^= targeted_bb

    pass


ROOK_LKUP_TBL = {}
for sqr in range(64):
    rook_tuple = arrangments("r", sqr)
    for blocker_bb_idx, blocker_bb_elem in enumerate(rook_tuple[0]):
        ROOK_LKUP_TBL[(sqr, blocker_bb_elem)
                      ] = rook_tuple[1][blocker_bb_idx]

BISH_LKUP_TBL = {}
for sqr in range(64):
    bish_tuple = arrangments("b", sqr)
    for blocker_bb_idx, blocker_bb_elem in enumerate(bish_tuple[0]):
        BISH_LKUP_TBL[(sqr, blocker_bb_elem)
                      ] = bish_tuple[1][blocker_bb_idx]


def draw_ending(col):

    clock = py.time.Clock()

    update_screen()
    text_surf = END_FONT.render(f"{col.title()} JUST WON!!", 1, (YELLOW))
    WIN.blit(text_surf, (WIDTH/2, HEIGHT/2))

    while 1:

        py.display.update()
        clock.tick(FPS)

        for event in py.event.get():
            if event.type == py.QUIT:
                sys.exit()


def main():

    # 0 = white, 1 = black
    turn = 0

    clock = py.time.Clock()

    fens = {
        # STARTING POSTION
        0: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",

        # FEN CASTLING CHECK (WORKS)
        1: "r3k2r/ppppqppp/8/8/6N1/8/PPPPPPPP/R3K2R w Kq - 0 1",

        # FEN ENPESANT CHECK (WORKS)
        2: "r3k2r/pp1pqp1p/8/2pP1Pp1/6N1/8/PPP1P1PP/R3K2R w Kq g6 0 1",

        # PROMOTION TEST AND ALSO SEE IF CAN DO IF PINNED (WORKS)
        3: "5qk1/K2P3r/8/8/7R/8/PPBPPPPP/RNBQ2N1 w - - 0 1",

        # CHECKING BISHOP MOVEMENT (WORKS)
        4: "7k/8/8/4B3/2b5/8/8/K7 w - - 1 1",

        # PUZZLE 1
        5: "5r2/8/1R6/ppk3p1/2N3P1/P4b2/1K6/5B2 w - - 0 1",

        # PUZZLE 2
        6: "2kr3r/pp3p2/2pP4/2q2bpp/4p3/4P1B1/PPP2PPP/R1Q2RK1 b - - 0 1",

        # PINNED (HORSE) (WORKS)
        7: "r1bqkbnr/ppp5/2n5/1B4p1/8/8/PPPPPPPP/RNBQK1NR b KQkq - 0 1",

        # PINNED (BISHOP) (WORKS)
        8: "r1bqkbnr/ppp5/2b5/1B4p1/8/8/PPPPPPPP/RNBQK1NR b KQkq - 0 1",

        # IN CHECK WITH PIN (HORSE) (WORKS)
        9: "rnbqkb1r/ppp2ppp/2n5/3p4/B7/8/8/3K1R2 w - - 0 1",

        # CHECK WITH PIN (BISHOP) (WORKS)
        10: "rnbqkb1r/ppp2ppp/2b5/8/B7/8/3P4/3K1R2 w - - 0 1",

        # CANT GET OUT OF CHECK WITH KING BUT CAN BLOCK (WORKS)
        11: "1nbqk3/ppppp3/8/3Q4/8/5r2/PPPPPPPP/RNB1KBNR w KQ - 0 1",

        # PAWN CHECKING THINGS (WORKS)
        12: "8/4k3/8/3P4/8/8/8/K7 w - - 0 1",

        # CASTLING (WORKS)
        13: "rnbqk2r/pppppppp/8/8/8/1Q6/8/R3K2R w Qkq - 0 1",

        # CASTLE THROUGH CHECK. CHECKER (WORKS)
        14: "1nbqkbnr/pppppppp/2r5/5N2/1P6/4BP2/P2PPPPP/R2QKBNR w QKqk - 0 1",

        # CASTLE OUT OF CHECK, CHECKER (WORKS)
        15: "1nbqkbnr/pppppppp/4r3/5N2/1P6/5P2/P1QB1PPP/R3KBNR w KQkq - 0 1",

        # CHECKMATE CHECKER (CHECKED WHERE ONLY PIECE THAT CAN BLOCK IS PINNED)
        # (WORKS)
        16: "4k3/R7/6r1/7B/3Q4/8/8/4K3 w - - 0 1",

        # HORSE CHECKING (WORKS)
        17: "1N1pkpNN/3prp2/r7/5N2/8/8/2N1P3/4KP2 w - - 0 1",

        # MORE INTRICATE TEST FOR CHECKMATE
        18: "rnbqkbn1/pp1prppp/8/8/4N3/8/PPPPPPPP/RNBQKB1R w KQq - 0 1",

        # double check craziness
        19: "4k1q1/6p1/p7/8/8/8/PPPPBPPP/RNBQR1KN w - - 0 1",

        # double check craziness BUT CANT MOVE KING
        20: "3rkr2/1q3pp1/p7/8/8/8/PPPPBPPP/RNBQR1KN w - - 0 1",
    }

    turn = fen_2_bb(fens[0])

    update_screen()

    sel_sqr = None
    legal_sqrs = None

    mouse_down = False

    while 1:

        if not mouse_down:

            if not legal_sqrs is None:
                mouse_pos = py.mouse.get_pos()
                for idx, elem in enumerate(GRIDS):
                    if elem.collidepoint(mouse_pos):
                        sqr_released = idx
                        break

                if sqr_released in legal_sqrs:

                    if update_bb(piece_col, piece_type, sel_sqr, sqr_released) == 1:
                        turn ^= 1
                    elif update_bb(piece_col, piece_type, sel_sqr, sqr_released) == "checkmate":
                        draw_ending(piece_col)

                update_screen()

            sel_sqr = None
            legal_sqrs = None

        if mouse_down:

            mouse_pos = py.mouse.get_pos()
            for idx, elem in enumerate(GRIDS):
                if elem.collidepoint(mouse_pos):
                    if sel_sqr is None:
                        sel_sqr = idx

                    piece_col, piece_type = give_piece(sel_sqr)
                    if not piece_type or not piece_col:
                        pass
                    else:
                        if legal_sqrs is None:

                            # ------------ turns -------------- #
                            # forgot how this works but it works (turn: 0 -> white, 1 -> black)
                            if ((turn == 0) ^ (piece_col == "white")):
                                legal_bb = 0
                            # ----------------------------------- #
                            else:
                                legal_bb = give_legal_moves(
                                    piece_col, piece_type, sel_sqr)

                            legal_sqrs = get_indices(legal_bb)

                        pickup(sel_sqr, piece_col, piece_type,
                               mouse_pos, legal_sqrs)
                    break

        # WIN.blit(TEST_SURF, (GRIDS[grid_idx].x, GRIDS[grid_idx].y))
        clock.tick(FPS)
        py.display.update()

        for event in py.event.get():
            if event.type == py.QUIT:
                sys.exit()

            elif event.type == py.MOUSEBUTTONDOWN:
                if event.button:  # Left mouse button
                    mouse_down = True

            elif event.type == py.MOUSEBUTTONUP:
                if event.button:  # Left mouse button
                    mouse_down = False


if __name__ == "__main__":
    main()

# THINGS I COULD DO TO OPTIMIZE:
# 1: keep track of where peices are so i dont have to loop over board to find them
# 2. keep track of all sqaures a colour attacks so i dont have to loop so much to detect a check
#    This will help with checkmate aswell (obviously)
# 3. Optimize prevoius attempt at using magic bitboards
# 4. Use a quicker langauge like C++
