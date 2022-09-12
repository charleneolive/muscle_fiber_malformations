# calculate branch angle: https://forum.image.sc/t/how-to-obtain-the-angle-of-a-skeletal-branch-point-in-skan/55065/2

# https://github.com/jni/skan/issues/78

# skeleton analysis: https://skeleton-analysis.org/getting_started.html 

# about determining threshold for canny edge: https://justin-liang.com/tutorials/canny/#suppression

# find parameters for canny edge: https://github.com/maunesh/opencv-gui-helper-tool

# double thresholding & edge tracking by hysteresis: https://github.com/hasbisevinc/Canny-Edge-Detection-Algorithm


import os
import cv2
import scipy 
import skimage
import glob
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
from skan import csr, draw
from scipy import stats
from matplotlib import collections
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
# from skan import Skeleton, summarize, branch_statistics
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import skeletonize, rectangle, disk, opening, closing, binary_dilation, remove_small_objects

# use the test
path = '../Results/2022-08-21/after_nms'
original_path = '../data/Images'
save_path = '../Results/2022-09-12'
groups = [f for f in glob.glob(os.path.join(path, '*')) if os.path.isdir(f)]


high_ratio = 0.2
low_ratio = 0.05
# in microns
border = 20
thres_min_size = 100 # i.e. 100 microns
length = 50 # in microns
resolution = 0.828640759202985 # microns per pixel, taken from metadata

def isolate_by_branch_type(image, graph_class, branch_data, branch_type_mapping, image_skeleton, branch_type, color, ax):
    new_image_skeleton = np.zeros((image_skeleton.shape[0], image_skeleton.shape[1]), dtype=bool)
    for ii in range(np.size(branch_data, axis=0)):
        if branch_data.iloc[ii,4] == branch_type_mapping[branch_type]: # branch type
            # add the branch
            for jj in range(np.size(graph_class.path_coordinates(ii), axis=0)):
                new_image_skeleton[int(graph_class.path_coordinates(ii)[jj,0]), int(graph_class.path_coordinates(ii)[jj,1])] = True

    image[new_image_skeleton] = np.array(color)
    ax.imshow(image)
    ax.axis('off')
    
    return ax

def plotSigLevel(x1, x2, gp1, gp2, ax):
    
    stat, p_value = stats.ttest_ind(gp1, gp2, equal_var=False, alternative='two-sided')
    mapping = ['trend', '*', '**']
    sig_value = None
    if p_value<=1.00e-01 and p_value>5.00e-02:
        sig_value = mapping[0]
    elif p_value <=5.00e-02 and p_value>1.00e-02:
        sig_value = mapping[1]
    elif p_value <=1.00e-02:
        sig_value = mapping[2]
        
    if sig_value is None:
        return
        
    max_value = 1.1*max(gp1.max(), gp2.max())
    height = 0.05*max(gp1.max(), gp2.max())
    y, h, col = max_value, height, 'k'
    ax.plot([x1,x1,x2,x2], [y,y+h,y+h,y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+0.9*h, sig_value, ha="center", va="bottom", color=col, fontsize=20)
    
def plot_graph(data_df, x, y):

    plt.figure(figsize=(15,10))
    ax = sns.boxplot(data = data_df, x=x, y=y, color = "skyblue", width = 0.5)
    ax = sns.scatterplot(data = data_df, x=x, y=y, color = "black", s=100)
    
    plotSigLevel(0, 1, data_df[data_df['group']=='Cardiotoxin'][y], data_df[data_df['group']=='PU + DFO'][y], ax)
    plotSigLevel(0, 2, data_df[data_df['group']=='Cardiotoxin'][y], data_df[data_df['group']=='PU + saline'][y], ax)
    plotSigLevel(1, 2, data_df[data_df['group']=='PU + DFO'][y], data_df[data_df['group']=='PU + saline'][y], ax)
    ax.set_xlabel(x,fontsize=30, labelpad=10)
    ax.set_ylabel(y,fontsize=30, labelpad=10)
    plt.xticks(fontsize= 25)
    plt.yticks(fontsize= 25)

    ax.set_title(y +' in different ' + x, fontsize=30)
    plt.savefig(os.path.join(save_path, x + '_' + y + '.png'))
    
    
def getWidth(imageArray):
    return len(imageArray[0])

def getHeight(imageArray):
    return len(imageArray)


def doubleThreshold(image, lowThreshold, highThreshold):
    image[np.where(image > highThreshold)] = 255
    image[np.where((image >= lowThreshold) & (image <= highThreshold))] = 75
    image[np.where(image < lowThreshold)] = 0
    return image

def edgeTracking(image):
    width = getWidth(image)
    height = getHeight(image)
    for i in range(0, height):
        for j in range(0, width):
            if image[i][j] == 75:
                if ((image[i+1][j] == 255) or (image[i - 1][j] == 255) or (image[i][j + 1] == 255) or (image[i][j - 1] == 255) or (image[i+1][j + 1] == 255) or (image[i-1][j - 1] == 255) or (image[i - 1][j + 1] == 255) or (image[i + 1][j - 1] == 255)):
                    image[i][j] = 255
                else:
                    image[i][j] = 0
    return image


def process_single_im(im, original_path, group_name):
    ## ------------ convert to skeleton ... ----------
    image = cv2.imread(im, cv2.IMREAD_GRAYSCALE) # edges detected by deep learning model
    diff = np.max(image) - np.min(image)
    t_low = np.min(image) + low_ratio * diff
    t_high = np.min(image) + high_ratio * diff

    ori_image = cv2.imread(os.path.join(original_path, group_name, os.path.splitext(os.path.basename(im))[0].replace('_ms', '')+'.jpg')) # original image
    corr_ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    scaling = image.shape[0]/ corr_ori_image.shape[0]

    corr_resolution = resolution / scaling


    temp_image = doubleThreshold(image.copy(), t_low, t_high)
    edge_image = edgeTracking(temp_image.copy())

    image_skeleton_temp = skeletonize(edge_image>0).astype(bool)

    # remove sides due to edge artifacts

    border_pixels = round(border/corr_resolution)
    image_skeleton = np.zeros((image_skeleton_temp.shape[0], image_skeleton_temp.shape[1]), dtype=bool)
    image_skeleton[border_pixels:-border_pixels, border_pixels:-border_pixels] = image_skeleton_temp[border_pixels:-border_pixels, border_pixels:-border_pixels]
    # ------- USING SKAN --------
    # https://github.com/jni/skan/issues/92
    graph_class = csr.Skeleton(image_skeleton)
    branch_data = csr.summarize(graph_class)

    ori_branch_data = branch_data.copy()

    # remove branches where euclidean_distance is 0, remove circular objects
    # branch_data = branch_data[branch_data['euclidean-distance']!=0]
    thres_min_size_pixels = thres_min_size/corr_resolution
    length_pixels = length/corr_resolution

    for ii in range(np.size(ori_branch_data, axis=0)):
        if ori_branch_data.loc[ii,'branch-distance'] < thres_min_size_pixels: # branch distance
            # grab NumPy indexing coordinates, ignoring endpoints
            integer_coords = tuple(
                graph_class.path_coordinates(ii)[1:-1].T.astype(int)
            )
            # remove the branch
            image_skeleton[integer_coords] = False

        elif ori_branch_data.loc[ii,'euclidean-distance'] < length_pixels: #euclidean distance
            # grab NumPy indexing coordinates, ignoring endpoints
            integer_coords = tuple(
                graph_class.path_coordinates(ii)[1:-1].T.astype(int)
            )
            # remove the branch
            image_skeleton[integer_coords] = False

    image_skeleton = skimage.morphology.remove_small_objects(image_skeleton, min_size=4, connectivity=2) # remove objects of pixels 3 and below - i.e. 3 junction-to-endpoints

    graph_class2 = csr.Skeleton(image_skeleton)
    branch_data2 = csr.summarize(graph_class2)
    
    # Remove any anomalies afterwards 
    for ii in range(np.size(branch_data2, axis=0)):
        if branch_data2.loc[ii,'euclidean-distance'] <2: #euclidean distance is 2
            
            integer_coords2 = tuple(
                graph_class2.path_coordinates(ii).T.astype(int)
            )
            # remove the branch
            image_skeleton[integer_coords2] = False

    graph_class3 = csr.Skeleton(image_skeleton)
    branch_data3 = csr.summarize(graph_class3)
    
    return image, corr_ori_image, corr_resolution, image_skeleton, graph_class3, branch_data3



total_stats = {k:[] for k in 
               ['image_name', 'group', 'num_edge_segments', 'num_edge_segments_per_area', 
                'median_tortuosity', 'median_edge_segment_distance', 
                'orientation_radians_std','orientation_radians_iqr',
               'endpoint-to-endpoint','junction-to-endpoint','junction-to-junction','isolated cycle']}
skeleton_dict = {}
image_dict = {}
ori_image_dict = {}
branch_df_dict = {}
orientation_stats = {os.path.basename(k):[] for k in groups}



for group_path in groups:
    save_skeleton_path = os.path.join(save_path, 'skeletons', os.path.basename(group_path))
    save_overlay_path = os.path.join(save_path, 'overlay', os.path.basename(group_path))
    pathlib.Path(save_skeleton_path).mkdir(exist_ok=True, parents=True)
    pathlib.Path(save_overlay_path).mkdir(exist_ok=True, parents=True)
    images_group = glob.glob(os.path.join(group_path, '*.png'))
    
    
    for im in images_group:
        group_name = os.path.basename(group_path)

        image, corr_ori_image, corr_resolution, result_skeleton3, graph_class3, branch_data3 = process_single_im(im, original_path, group_name)
        corr_ori_image_resized = cv2.resize(corr_ori_image, (image.shape[1], image.shape[0]))
        area = (corr_ori_image.shape[0]*resolution) * (corr_ori_image.shape[1]*resolution) # total area in microns
        ## ------------ get relevant biomarkers ... ----------
        num_branches = len(branch_data3)
        coords_src = branch_data3[['coord-src-0', 'coord-src-1']].to_numpy()
        coords_dst = branch_data3[['coord-dst-0', 'coord-dst-1']].to_numpy()

        # find vector describing orientation of branch
        vectors = coords_dst - coords_src
        # find the angle of the vector (in the range [-π, π])
        angle = np.arctan2(vectors[:, 0], vectors[:, 1])
        # add the orientation of the vector and add it to data frame
        # (in the range [0, π], because branches are not directed)
        
        branch_data3['orientation-radians'] = angle % np.pi
        # or if you prefer
        branch_data3['orientation-degrees'] = np.degrees(angle) % 180
        branch_data3['tortuosity'] = branch_data3['branch-distance']/ branch_data3['euclidean-distance']
        median_tortuosity = branch_data3[branch_data3['tortuosity']!=np.inf]['tortuosity'].median()

        data_orientation = np.array(branch_data3['orientation-radians'].tolist())
        # print(data_orientation)
        q3, q1 = np.percentile(data_orientation, [75 ,25])
        iqr = q3 - q1
        num_branches_per_area = num_branches/area
        branch_dist_microns = branch_data3['branch-distance'].median() * corr_resolution
        

        ## ------------ add to stats & record... ----------
        
        total_stats['image_name'].append(os.path.basename(im))
        total_stats['group'].append(os.path.basename(group_path))
        total_stats['num_branches'].append(num_branches)
        total_stats['num_branches_per_area'].append(num_branches_per_area)
        total_stats['median_tortuosity'].append(median_tortuosity)
        total_stats['median_branch_distance'].append(branch_dist_microns)
        total_stats['orientation_radians_std'].append(branch_data3['orientation-radians'].std())
        total_stats['orientation_radians_iqr'].append(iqr)
        total_stats['endpoint-to-endpoint'].append(len(branch_data3[branch_data3['branch-type']==0]))
        total_stats['junction-to-endpoint'].append(len(branch_data3[branch_data3['branch-type']==1]))
        total_stats['junction-to-junction'].append(len(branch_data3[branch_data3['branch-type']==2]))
        total_stats['isolated cycle'].append(len(branch_data3[branch_data3['branch-type']==3]))
        
        branch_df_dict[os.path.basename(im)] = branch_data3
        skeleton_dict[os.path.basename(im)] = result_skeleton3
        orientation_stats[os.path.basename(group_path)].extend(branch_data3['orientation-radians'].tolist())
        ori_image_dict[os.path.basename(im)] = corr_ori_image_resized
        
        fig, ax = plt.subplots(figsize=(20,20))
        branch_type_mapping = {'endpoint-to-endpoint':0, 'junction-to-endpoint':1, 'junction-to-junction': 2, 'isolated cycle': 3}

        new_ax = isolate_by_branch_type(corr_ori_image_resized, graph_class3, branch_data3, branch_type_mapping, result_skeleton3, 'endpoint-to-endpoint', (255, 0, 0), ax)
        new_ax = isolate_by_branch_type(corr_ori_image_resized, graph_class3, branch_data3, branch_type_mapping, result_skeleton3, 'junction-to-endpoint', (0, 255, 0), ax)
        new_ax = isolate_by_branch_type(corr_ori_image_resized, graph_class3, branch_data3, branch_type_mapping, result_skeleton3, 'junction-to-junction', (0, 0, 255), ax)
        new_ax = isolate_by_branch_type(corr_ori_image_resized, graph_class3, branch_data3, branch_type_mapping, result_skeleton3, 'isolated cycle', (255, 0, 255), ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_overlay_path, 'overlay_'+os.path.basename(im)))
        plt.close()
        
        cv2.imwrite(os.path.join(save_skeleton_path, 'skeleton_'+os.path.basename(im)), np.uint8(result_skeleton3*255))
        fig, ax = plt.subplots(figsize=(20,20))
        draw.overlay_skeleton_2d(corr_ori_image_resized, result_skeleton3, dilate=0, axes=ax, color=(1,1,1))
        plt.savefig(os.path.join(save_skeleton_path, 'overlay_'+os.path.basename(im)))
        plt.close()
        
        
data_df = pd.DataFrame(total_stats)

data_df.to_csv(os.path.join(save_path, 'total_stats.csv'), index=False)

plot_graph(data_df, 'group', 'median_tortuosity')
plot_graph(data_df, 'group', 'orientation_radians_std')
plot_graph(data_df, 'group', 'orientation_radians_iqr')
plot_graph(data_df, 'group', 'num_branches_per_area')
plot_graph(data_df, 'group', 'num_branches')
plot_graph(data_df, 'group', 'median_branch_distance')