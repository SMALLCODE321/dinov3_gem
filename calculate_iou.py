 
def IoU_calculate(result, ground_truth):  
    x1_left, y1_top, x1_right, y1_bottom = result
    x2_left, y2_top, x2_right, y2_bottom = ground_truth
    
    # Check if there is no intersection
    if x1_right < x2_left or x1_left > x2_right or y1_bottom < y2_top or y1_top > y2_bottom:  
        return 0.0  # No intersection
    
    # Calculate the intersection boundaries correctly
    x_intersect_left = max(x1_left, x2_left)
    x_intersect_right = min(x1_right, x2_right)
    y_intersect_top = max(y1_top, y2_top)  
    y_intersect_bottom = min(y1_bottom, y2_bottom)
    
    # Ensure that the intersection area is valid
    if x_intersect_right > x_intersect_left and y_intersect_bottom > y_intersect_top:
        intersect_area = (x_intersect_right - x_intersect_left) * (y_intersect_bottom - y_intersect_top)
    else:
        intersect_area = 0.0
    
    # Calculate the area of the two rectangles
    rect1_area = (x1_right - x1_left) * (y1_bottom - y1_top)
    rect2_area = (x2_right - x2_left) * (y2_bottom - y2_top)
    
    # Calculate the union area
    union_area = rect1_area + rect2_area - intersect_area
    
    # Return the IoU
    return intersect_area / union_area 


def file_IoU_calculate(result_fname, ground_truth_fname):  
    dic_results = {}
    with open(result_fname, mode='r', encoding='utf-8') as file: 
        lines = file.readlines() 
    for line in lines:
        line = line.strip()
        data = line.split(' ')
        dic_results[data[0]] = (data[1], [int(data[2]), int(data[3]), int(data[4]), int(data[5])])
   
    
    dic_ground_truth = {}
    with open(ground_truth_fname, mode='r', encoding='utf-8') as file: 
        lines = file.readlines() 
    for line in lines:
        line = line.strip()
        data = line.split(' ')
        dic_ground_truth[data[0]] = (data[1], [int(data[2]), int(data[3]), int(data[4]), int(data[5])])
    
    dic_topk = {}
   
            
    all_iou = 0
    counter = 0
    for key, value in dic_results.items(): 
        if value[0] == dic_ground_truth[key][0]:
            counter += 1
            iou = IoU_calculate(dic_results[key][1], dic_ground_truth[key][1]) 
            all_iou += iou
            if iou < 0.9:
                print(f'{key}--{value[0]}------IoU:{iou}')
        else:
            print(f'{key}--{value[0]}------gt:{dic_ground_truth[key][0]}')
            print(f'{key}--{value[1]}------gt:{dic_ground_truth[key][1]}')
            
    acc = counter/len(dic_ground_truth) # accuracy of the db image search
    m_iou = all_iou/len(dic_ground_truth)
    print(all_iou)
    print(len(dic_ground_truth))
    return m_iou, acc
    
   
if __name__ == '__main__': 
    result_fname = './exps/exp_train3_salad_672_980_fea_log7/32-RESULT_test_large_flann.txt'
    result_topk_fname = './exps/exp_train3_salad_672_980_fea_log7/32-RESULT_test_large_flann_topk.txt' 
    truth_fname = './exps/ground_truth_large.txt'

    m_iou, acc = file_IoU_calculate(result_fname, truth_fname, result_topk_fname) 
    print(f"mIoU:{m_iou}    Acc: {acc}")