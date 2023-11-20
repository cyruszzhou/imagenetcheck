import math
from itertools import combinations_with_replacement
import random
import torch
from torch.autograd import Function

random.seed(42)

class Quantize(Function):
    @staticmethod
    def forward(ctx, w, p, qf):
        return quantize(w, p, qf)
        
    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None

def quantize(w, p, qf):
    mag_w = torch.max(torch.abs(w)).item()
    norm2 = general_quantize(w, p, qf, 0, mag_w/2)
    return norm2

# class QA(Function):
#     @staticmethod
#     def forward(ctx, x, p):
#         return scaledQuanitze(x,p)

#     @staticmethod
#     def backward(ctx, gradOutput):
#         return None, None


def inputQuantizeClamp(x, p):
    y = torch.clamp(x, min=-2, max=2)

    y = torch.add(y, 2.0)
    y = torch.mul(y, (2 ** p) )
    y = torch.round(y)

    adj_num = 2 - (1 / (2**(p-1)) )

    y = torch.mul(y, (adj_num / 16))
    y = torch.sub(y, (adj_num / 2))

    return y

## gen quantize
def general_quantize(w, p, qf, offset, delta):
    # print("gq device", p.device)
    # p.to('cuda')
    # print(p.device)
    # newp = p.to("cuda")
    alpha = 2**(p-2) / delta
    beta = delta*(2-2**(1-p)) + offset

    
    # alpha = alpha.to("cuda")
    # beta = beta.to("cuda")

    # print("alpha, w, beta devices", alpha.device, w.device, beta.device)
    w_scaled = alpha * (w + beta)
    w_rounded = torch.round(w_scaled)
    w_rounded = torch.clamp(w_rounded, min=0)
    w_rounded = torch.minimum(w_rounded, 2**p-1)
    w_quant = w_rounded/alpha - beta

    return w_quant

 ###### Standard 4,2,1 ops #####
# standard sub-4 bit quanitization
def no3bit(val):
    four = (val > 3.0) * 4.0
    two = ((val <= 3.0) & (val > 1.5)) * 2.0
    one = (val <= 1.5) * 1.0

    return four + two + one

# sub-4bit qunatization upgrading more values to 4-bit's
def no3bitAccurate(val):
    four = (val > 2.0) * 4.0
    two = ((val <= 2.0) & (val > 1.5)) * 2.0
    one = (val <= 1.5) * 1.0

    return four + two + one

# overall function to quanitize to [1,2,4], option to focus on 
# accuracy or size
def no_3(tensor, accurate=False):
    if accurate:
        return no3bitAccurate(tensor)
    else:
        return no3bit(tensor)   

 ###### Standard 4,2,1 ops #####  

## creates a p Tensor based on inputs
# def createPTensor(w, s, set_prec=None, spm='None', sg2=None, norm=True):
#     if norm:
#         scaling = max(1e-6, min(1, torch.abs(w).max().item()))

#         # get the float representation of the precision
#         float_p = 1-torch.log2(torch.sigmoid(s)/scaling)
#     else:
#         float_p = 1-torch.log2(torch.sigmoid(s))

#     if spm == 'None':
#         p = no_3(float_p)

#     else:
#         p = createPTensorHelp(set_prec, float_p, spm, sg2)
#         p.to('cuda')

#     return p

def createPTensor(w, s, set_prec=None, spm='None', norm=True):
    if norm:
        scaling = max(1e-6, min(1, torch.abs(w).max().item()))

        # get the float representation of the precision
        float_p = 1-torch.log2(torch.sigmoid(s)/scaling)
    else:
        float_p = 1-torch.log2(torch.sigmoid(s))

    if spm == 'None' or set_prec == None:
        p = no_3(float_p)
        # print("generated p")
        # print(p)
        # print("inp", p.shape)

    if spm != 'None':
        if set_prec == None:
            # generate set_prec
            # print("should be in here!")
            list = p.tolist()
            # print("list", list)

            if type(list) == float:
                set_prec = [list]
                # set_prec = get421Precs(set_prec)
                # set_prec = [(1, set_prec[0]), (2, set_prec[1]), (4, set_prec[2])]
                set_prec = countPrecs(set_prec)

            else:
                list = list[0]

                for i in range(len(list)):
                    list[i] = list[i][0][0]
                
                set_prec = countPrecs(list)

                # print("new list", list)
                # set_prec = set_prec = get421Precs(list)
                # set_prec = [(1, set_prec[0]), (2, set_prec[1]), (4, set_prec[2])]

            # print(set_prec)


        

        p = createPTensorHelp(set_prec, float_p, spm)
        # p.to('cuda')

    return p

# counts the precisions. might be redundant func
def countPrecs(list):
    four, two, one = 0, 0, 0

    for x in list:
        if x == 1:
            one += 1

        if x == 2:
            two += 1

        if x == 4:
            four += 1

    return [(1, one), (2, two), (4, four)]


# simple function that returns the precisions frequencies of 
# 4,2,1. returned as [prec_freq_1, prec_freq_2, prec_freq_4]
def get421Precs(alist):
    four, two, one = 0, 0, 0

    for x in alist:
        if x[0] == 4:
            four = x[1]

        if x[0] == 2:
            two = x[1]

        if x[0] == 1:
            one = x[1]

    return [one, two, four]  

## returns p vector from set precisions, float_p 
def createPTensorHelp(set_prec, float_p, spm):

    # intitalize variables
    ans = float_p.clone()
    val_with_i_arr = []


    # edge case that appears sometimes.
    if len(ans.shape) == 0:
        # print(set_prec)
        ans.copy_(torch.tensor(set_prec[0][0]))
        return ans



    # add s value and index tuples to a list so they may be sorted and therefore ranked
    for i in range( ans.shape[1] ):
        val_with_i_arr.append( (ans[0][i][0][0].item(), i) )

    # set the largest s valeus to the highest priority (set by iterating from 0 -> n-1)
    val_with_i_arr.sort(reverse=True)

    # ensure the order is (1, 2, 4). can be made more efficient by doing this once
    set_prec.sort()



    # adjust the precisions to desired values
    if spm == 'vector_gen_precs' or spm == 'vector':
        set_prec = simpleuniform(set_prec)




    elif spm == 'state_gen_precs' or spm == 'state':
        set_prec = simplestate(set_prec)


    elif 'pattern' in spm:
        set_prec = patternQuantize(set_prec, spm)



        list_of_precs = set_prec[0]
        # with open('cyruspatternnums.txt', 'a') as file:
        #     print("layer_pattern_num:", len(list_of_precs), file=file)

        one, two, four = 0, 0, 0

        for list in list_of_precs:
            four += list[2]; two += list[1]; one += list[0]

        set_prec = [(1, one), (2, two), (4, four)]


    else:
        raise ValueError("Invalid set precision mode given!")


    nums = get421Precs(set_prec)

    # set precisons based on s ranking
    for four in range( nums[2] ):
        if four >= ans.shape[1]:
            break

        index = val_with_i_arr[four][1]
        ans[0][ index ][0][0] = torch.tensor(4.0)

    for two in range( nums[1] ):
        if (two + nums[2]) >= ans.shape[1]:
            break
        index = val_with_i_arr[nums[2] + two][1]
        ans[0][ index ][0][0] = torch.tensor(2.0)

    for one in range( nums[0] ):
        if (one + nums[1] + nums[2]) >= ans.shape[1]:
            break
        index = val_with_i_arr[nums[2] + nums[1] + one][1]
        ans[0][ index ][0][0] = torch.tensor(1.0)

    return ans


## simpler is better.
## this function ensures uniform precisions along the vector
def simpleuniform(precs):
    four, two, one = 0,0,0

    for prec_count in precs:
        # print(prec_count)
        if prec_count[0] == 4:
            four = prec_count[1]

        if prec_count[0] == 2:
            two = prec_count[1]

        if prec_count[0] == 1:
            one = prec_count[1]

    while (four % 32) != 0 and two > 0:
        four += 1; two -= 1

    while (four % 32) != 0 and one > 0:
        four += 1; one -= 1

    while (two % 32) != 0 and one > 0:
        two += 1; one -= 1

    return ((1, one), (2, two), (4, four))


## this function ensures uniform precisions for each state
def simplestate(precs):
    four, two, one = 0,0,0

    for prec_count in precs:
        if prec_count[0] == 4:
            four = prec_count[1]

        if prec_count[0] == 2:
            two = prec_count[1]

        if prec_count[0] == 1:
            one = prec_count[1]

    while (four % 4) != 0 and two > 0:
        four += 1; two -= 1

    while (four % 4) != 0 and one > 0:
        four += 1; one -= 1

    while (two % 8) != 0 and one > 0:
        two += 1; one -= 1

    return ((1, one), (2, two), (4, four))


def patternQuantize(precs, spm='pattern_4'):

    patterns_4 = [(128,0,0),(0,64,0),(0,0,32),(32, 64,8)]
    patterns_45 = [(0, 0, 32), (0, 8, 28), (0, 16, 24), (0, 24, 20), (0, 32, 16), (0, 40, 12), 
        (0, 48, 8), (0, 56, 4), (0, 64, 0), (16, 0, 28), (16, 8, 24), (16, 16, 20), (16, 24, 16), 
        (16, 32, 12), (16, 40, 8), (16, 48, 4), (16, 56, 0), (32, 0, 24), (32, 8, 20), (32, 16, 16),
        (32, 24, 12), (32, 32, 8), (32, 40, 4), (32, 48, 0), (48, 0, 20), (48, 8, 16), (48, 16, 12),
        (48, 24, 8), (48, 32, 4), (48, 40, 0), (64, 0, 16), (64, 8, 12), (64, 16, 8), (64, 24, 4),
        (64, 32, 0), (80, 0, 12), (80, 8, 8), (80, 16, 4), (80, 24, 0), (96, 0, 8), (96, 8, 4), (96, 16, 0),
        (112, 0, 4), (112, 8, 0), (128, 0, 0)]

    patterns_16 = None


    def split_tuple_randomly(N):
        splitby = 128
        x, y, z = N
        result = []

        while x > 0 or y > 0 or z > 0:
            temp_x, temp_y, temp_z = 0, 0, 0
            current_sum = 0

            while current_sum < splitby:
                choice = random.choice(['x', 'y', 'z'])
                
                if choice == 'x' and x > 0 and current_sum + 1 <= splitby:
                    temp_x += 1
                    x -= 1
                elif choice == 'y' and y > 0 and current_sum + 1 <= splitby:
                    temp_y += 1
                    y -= 1
                elif choice == 'z' and z > 0 and current_sum + 1 <= splitby:
                    temp_z += 1
                    z -= 1

                current_sum = temp_x + temp_y + temp_z

                # Break if no more additions are possible
                if x + y + z == 0 or current_sum == 128:
                    break

            result.append((temp_x, temp_y, temp_z))

        return result

    def find_tuples(gamma):
        solutions = []

        # Check all potential combinations for n1, n2, and n4.
        for n1 in range(129):  # Since max value for n^i_1 in equation 1 is 128
            for n2 in range(65):  # Since max value for 2n^i_2 in equation 1 is 128
                for n4 in range(33):  # Since max value for 4n^i_4 in equation 1 is 128

                    # Check if the combination satisfies all the equations
                    if (4*n4 + 2*n2 + n1 == 128 and
                        math.ceil(n4/4) + math.ceil(n2/8) + math.ceil(n1/16) == 8 and
                        n4 + n2 + n1 >= 32 and
                        (n4 + n2 + n1) % gamma == 0):
                        
                        solutions.append((n1, n2, n4))

        return solutions

    ## parameters
    GAMMA = 4

    if spm == 'pattern_4':
        PATTERNS = patterns_4
    
    elif spm == 'pattern_45':
        PATTERNS = patterns_45

    else:
        raise ValueError("Bad pattern number passed to pattern_quantize!")


    promotion_penalties = [1.0,1.0,1.0]
    demotion_penalties = [1.0,1.0,1.0]

    ## best approximation section
    def objective_value_approximation(combination, N):
        total = [0, 0, 0]
        for tup in combination:
            for i in range(3):
                total[i]+=tup[i]

        value=0

        for i in range(3):
            if total[i] > N[i]:
                value += promotion_penalties[i] * (total[i] - N[i])
            elif N[i] > total[i]:
                value += demotion_penalties[i] * (N[i] - total[i])
        
        return value

    def sum_combination(combination):
        total = [0, 0, 0]
        for tup in combination: 
            for i in range(3):
                total[i] += tup[i]
        return tuple(total)


    def calc_avg_prec(combination):
        total = [0, 0, 0]
        
        for tup in combination: 
            for i in range(3):
                total[i] += tup[i]

        return (1*total[0]+2*total[1]+4*total[2])/(total[0]+total[1]+total[2])     

    def find_best_approximate(N):
        best_combination = None
        best_value = float('inf')

        for r in range(1, 11):
            for combination in combinations_with_replacement(PATTERNS, r):
                if sum(sum_combination(combination)) >= sum(N):
                    value = objective_value_approximation(combination, N)
                    if value < best_value:
                        best_combination = combination
                        best_value = value

        return best_combination, best_value


    ## upper bound section
    def objective_value_upperbound(combination):
        return len(combination)


    def sum_combination(combination):
        total = [0, 0, 0]
        for tup in combination: 
            for i in range(3):
                total[i] += tup[i]
        return tuple(total)


    def find_best_upperbound(N):
        # print(f"N is {N}")
        num_channel = sum(N)
        # print(f"num_channel is {sum(num_channel)}")
        result = []
        best_value = 0
        if num_channel > 128:
            best_combination = []

            split_tuple_list = split_tuple_randomly(N)
            for tpl in split_tuple_list:
                tmp_best_combination, tmp_best_value = _find_best_upperbound(tpl)
                for combination in tmp_best_combination:
                    best_combination.append(combination)
                best_value += tmp_best_value
        else:
            best_combination, best_value = _find_best_upperbound(N)

        return best_combination, best_value

    def _find_best_upperbound(N):

        best_combinations = []
        best_combination = None
        best_value = float('inf')

        GOTIT = False

        r = 0

        while True:
            for combination in combinations_with_replacement(PATTERNS, r):
                if sum(sum_combination(combination)) >= sum(N):
                    if sum_combination(combination)[2] >= N[2]:
                        if sum_combination(combination)[2]+sum_combination(combination)[1] >= N[2]+N[1]:
                            GOTIT = True
                            value = objective_value_upperbound(combination)
                            if value == best_value:
                                best_combinations.append(combination)      
                            if value < best_value:
                                best_combinations=[combination]
                                best_value = value

            if GOTIT:
                break
            r += 1

                
    
        best_value = 0
        for combination in best_combinations:
            value = calc_avg_prec(combination)
            if value > best_value:
                best_combination = combination
                best_value = value

        return best_combination, best_value

    ### actual calculation
    four, two, one = 0,0,0

    for prec_count in precs:
        if prec_count[0] == 4:
            four = prec_count[1]

        if prec_count[0] == 2:
            two = prec_count[1]

        if prec_count[0] == 1:
            one = prec_count[1]

    N = (one, two, four)

    # print("###### - Beginning Upperbound Search - #####")

    best_upper_bound = find_best_upperbound(N)

    with open('patterns_res_45.txt', 'a') as file:
        print("new pattern:", best_upper_bound, file=file)

    # print("###### - Concluded Upperbound Search - #####")
    return best_upper_bound


## other functions

# this is a function to return a tensor full of a given precision.
# it takes a tensor with the input channels filled with their precieions 
# and multipleis it by the other 3 dimensions ot generate a full matrix with 
# the precisions for each parameter.
def generate_p_tensor(input_precs, shape):
    input_fm_shape = (shape[2], shape[3])

    input_channels = torch.tensor([])
    for prec in input_precs:
        fm = torch.full(input_fm_shape, prec)
        input_channels = torch.cat((input_channels, fm), dim=0)

    full_tensor = input_channels(shape[0], 1, 1, 1)
    return full_tensor