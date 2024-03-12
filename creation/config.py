
"""
Data structure format
### One object
"data/one_object/initial_img/0~9/0~20" # 180
    "color_idx_rand_idx"
    "0/0"
    "0/19"
    "1_0"
"data/one_object/final_img/0~8" # 9 * 20 * 5
    "{init_color}_{rand_idx}_{direction}"
    "0/0_left"
    "0/0_right"
    "0/0_front"
    "1/0_front"
    "1/20_front"


### two objects
"data/two_objects/initial_img/0~72/0~20"
    "data/two_objects/initial_img/color_idx/rand_idx"
"data/two_objects/final_img/0~72"
    "{init_color_idx}_{init_rand_idx}_{color_1}_{direction}_{color2}"
    "0/0_red_left_blue"
    "0/1_red_left_blue"
    "0/20_red_left_blue"
    "0/0_red_right_blue"

"""



"""
Color map for different cubes / spheres.
"""
color_maps = {
    "red": (1, 0, 0),
    "blue": (0, 0, 1),
    "green": (0, 1, 0),
    "white": (1, 1, 1),
    "black": (0, 0, 0),
    "yellow": (1, 1, 0),
    "orange": (1, 0.5, 0),
    "purple": (0.5, 0, 0.5),
    "gray": (0.5, 0.5, 0.5),
}


"""
YCB object's size when it hits the ground at scale=1
It is used to determine the height of the object when 
it is placed on top of another.
"""
ycb_heights = {
    "002_master_chef_can": 0.07512415945529938,
    "003_cracker_box": 0.1150389239192009,
    "004_sugar_box": 0.09415584057569504,
    "005_tomato_soup_can": 0.05430442467331886,
    "006_mustard_bottle": 0.10356223583221436,
    "007_tuna_fish_can": 0.018264634534716606,
    "008_pudding_box": 0.02185451239347458,
    "009_gelatin_box": 0.016306713223457336,
    "010_potted_meat_can": 0.04430695250630379,
    "011_banana": 0.01998417265713215,
    "012_strawberry": 0.024883262813091278,
    "013_apple": 0.03828830644488335,
    "014_lemon": 0.028259683400392532,
    "015_peach": 0.0320853628218174,
    "016_pear": 0.03491714224219322,
    "017_orange": 0.038549233227968216,
    "018_plum": 0.028333280235528946,
    "019_pitcher_base": 0.1318930983543396,
    "021_bleach_cleanser": 0.1359364092350006,
    "022_windex_bottle": 0.14431481063365936,
    "024_bowl": 0.029312150552868843,
    "025_mug": 0.04315704479813576,
    "026_sponge": 0.010615101084113121,
    "028_skillet_lid": 0.04092275723814964,
    "029_plate": 0.015545567497611046,
    "030_fork": 0.00831411499530077,
    "031_spoon": 0.01119140349328518,
    "032_knife": 0.008370372466742992,
    "033_spatula": 0.018117250874638557,
    "035_power_drill": 0.0315856896340847,
    "036_wood_block": 0.11121562868356705,
    "037_scissors": 0.008466053754091263,
    "038_padlock": 0.015689071267843246,
    "040_large_marker": 0.010395342484116554,
    "042_adjustable_wrench": 0.008039338514208794,
    "043_phillips_screwdriver": 0.018805965781211853,
    "044_flat_screwdriver": 0.018605656921863556,
    "048_hammer": 0.01772714965045452,
    "050_medium_clamp": 0.01447580847889185,
    "051_large_clamp": 0.020444361492991447,
    "052_extra_large_clamp": 0.020130787044763565,
    "053_mini_soccer_ball": 0.06535003334283829,
    "054_softball": 0.05028996244072914,
    "055_baseball": 0.03869074583053589,
    "056_tennis_ball": 0.03524678200483322,
    "057_racquetball": 0.030031446367502213,
    "058_golf_ball": 0.022536057978868484,
    "059_chain": 0.01456491183489561,
    "061_foam_brick": 0.027590259909629822,
    "062_dice": 0.009448330849409103,
    "063-a_marbles": 0.03836605325341225,
    "063-b_marbles": 0.018574174493551254,
    "065-f_cups": 0.0387650690972805,
    "065-e_cups": 0.03833508864045143,
    "065-j_cups": 0.04084935411810875,
    "065-h_cups": 0.041207268834114075,
    "065-a_cups": 0.03342254087328911,
    "065-g_cups": 0.03956034407019615,
    "065-i_cups": 0.04150034859776497,
    "065-d_cups": 0.037895772606134415,
    "065-c_cups": 0.03583201393485069,
    "065-b_cups": 0.03461601957678795,
    "070-a_colored_wood_blocks": 0.08797957003116608,
    "070-b_colored_wood_blocks": 0.01401900127530098,
    "071_nine_hole_peg_test": 0.022982221096754074,
    "072-a_toy_airplane": 0.09603213518857956,
    "072-e_toy_airplane": 0.03568337857723236,
    "072-c_toy_airplane": 0.03595852106809616,
    "072-d_toy_airplane": 0.035807669162750244,
    "072-b_toy_airplane": 0.03229799121618271,
    "073-d_lego_duplo": 0.023365899920463562,
    "073-f_lego_duplo": 0.023438209667801857,
    "073-e_lego_duplo": 0.023555194959044456,
    "073-a_lego_duplo": 0.013218606822192669,
    "073-c_lego_duplo": 0.012895331718027592,
    "073-b_lego_duplo": 0.02339627407491207,
    "073-g_lego_duplo": 0.03246751427650452,
    "077_rubiks_cube": 0.03088015504181385
}