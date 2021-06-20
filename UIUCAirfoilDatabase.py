import requests
import re
import numpy as np
import time
import os
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

def rotatePts(pts, angle):
    angle = np.radians(angle)
    mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.einsum('ij,jk->ik',pts,mat)

def ptDist(pt1, pt2):
    return np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

def remDupPts(pts):
    rm_idx = []
    for i in range(1,pts.shape[0]):
        if pts[i,0] == pts[i-1,0] and pts[i,1] == pts[i-1,1]:
            rm_idx.append(i)
    pts = pts[[x for x in range(pts.shape[0]) if x not in rm_idx],:]
    return pts

def query_url(url):
    delay=10
    done = False
    while not done:
        res = requests.get(url)
        if res.status_code == 200:
            done = True
            return res
        elif res.status_code == 404:
            raise Exception('Not Found')
        else:
            print('Warning: Request timeout... waiting {}s.'.format(delay))
            time.sleep(delay)

def pullUIUCAirfoilData(reNum, ncrit):
    if ncrit not in [5,9]:
        raise Exception('Please choose an ncrit number that is available in the database')
    if reNum not in [50000, 100000, 200000, 500000, 1000000]:
        raise Exception('Please choose an Reynolds number that is available in the database')

    url = 'http://airfoiltools.com/search/airfoils'

    links = re.findall(r'href="/airfoil/details.*?">',query_url(url).text)
    for l in links:
        has_data = True
        url_airfoil = 'http://airfoiltools.com' + l[6:-2]
        print('Downloading Airfoil: {}'.format(url_airfoil))
        page_text = query_url(url_airfoil).text
        if 'The dat file is in Lednicer format' in page_text:
            fmt = 'Lednicer'
        elif 'The dat file is in Selig format' in page_text:
            fmt = 'Selig'
        else:
            raise Exception('dat file format unknown.')
        
        # get airfoil geometry
        url = re.findall(r'href="https://m-selig.ae.illinois.edu/ads/coord/.*?.dat', page_text)
        if len(url)==0:
            has_data = False
        else:
            url = url[0][6:]
            txt = query_url(url).text
            txt = txt.splitlines()
            name = txt[0]
            name = name.replace(' ', '_')
            name = name.replace('/', '_')
            name = name.replace(':', '_')
            name = name.replace('.', '_')
            name = name.replace('*', '_')
            if fmt == 'Lednicer':
                pts = txt[3:]
                nums = txt[1].split()
                nums = [int(float(n)) for n in nums]
            elif fmt == 'Selig':
                pts = txt[1:]
            pts = [p for p in pts if len(p) > 0]
            num_pts = 0
            arr = []
            for p in pts:
                temp = p.split()
                try:
                    temp = [float(t) for t in temp]
                except ValueError:
                    temp = []
                if len(temp) > 0:
                    arr.append(temp)
                    num_pts += 1
            arr = np.array(arr)

            if fmt == 'Lednicer':
                idx = np.concatenate([np.arange(num_pts-1,num_pts-nums[1]-1,-1), np.arange(1,nums[0],1)])
                arr = arr[idx,:]
            elif fmt == 'Selig':
                arr = arr[::-1]

        # get airfoil xfoil predicitons
        if ncrit == 9:
            polar_url = re.findall(r'href="/polar/details\?polar=xf-.*?'+str(reNum)+r'">', page_text)
        elif ncrit == 5:
            polar_url = re.findall(r'href="/polar/details\?polar=xf-.*?'+str(reNum)+r'-n5">', page_text)
        if len(polar_url) == 0:
            has_data = False
        else: 
            text = query_url('http://airfoiltools.com' + polar_url[0][6:-2]).text
            polar_text_url = re.findall(r'href="/polar/text\?polar=xf-.*?">', text)[0]
            polar_text = query_url('http://airfoiltools.com' + polar_text_url[6:-2]).text
            temp = '------ -------- --------- --------- -------- -------- --------'
            idx = polar_text.index(temp)
            polar_text = polar_text[idx+len(temp)+1:]
            polar_text = polar_text.splitlines()
            polar_arr = []
            for i, p in enumerate(polar_text):
                try:
                    polar_arr.append([float(x) for x in p.split()])
                except ValueError:
                    pass
            polar_arr = np.array(polar_arr)

        if has_data:
            if not os.path.isdir('UIUC/csv/'):
                os.mkdir('UIUC/csv/')
            if not os.path.isdir('UIUC/dat/'):
                os.mkdir('UIUC/dat/')
            if not os.path.isdir('UIUC/polars/'):
                os.mkdir('UIUC/polars/')
            if not os.path.isdir('UIUC/polars/uiuc_re{}_ncrit{}/'.format(str(reNum), ncrit)):
                os.mkdir('UIUC/polars/uiuc_re{}_ncrit{}/'.format(str(reNum), ncrit))
            np.savetxt('UIUC/csv/'+name+'.csv', arr, delimiter=',')
            np.savetxt('UIUC/dat/'+name+'.dat', arr, delimiter=' ',fmt='%f')
            np.savetxt('UIUC/polars/uiuc_re{}_ncrit{}/{}.csv'.format(str(reNum), ncrit, name), polar_arr,  delimiter=',', header='alpha, Cl, Cd, Cdp, Cm, Top_Xtr, Bot_Xtr', comments='')

def splineQuery(cv,u,steps=100,projection=True):
    #https://stackoverflow.com/questions/34941799/querying-points-on-a-3d-spline-at-specific-parametric-values-in-python
    ''' Brute force point query on spline
        cv     = list of spline control vertices
        u      = list of queries (0-1)
        steps  = number of curve subdivisions (higher value = more precise result)
        projection = method by wich we get the final result
                     - True : project a query onto closest spline segments.
                              this gives good results but requires a high step count
                     - False: modulates the parametric samples and recomputes new curve with splev.
                              this can give better results with fewer samples.
                              definitely works better (and cheaper) when dealing with b-splines (not in this examples)

    '''
    u = np.clip(u,0,1) # Clip u queries between 0 and 1

    # Create spline points
    samples = np.linspace(0,1,steps)
    tck,u_=interpolate.splprep(cv.T,s=0.0)
    p = np.array(interpolate.splev(samples,tck)).T   

    # Approximate spline length by adding all the segments
    p_= np.diff(p,axis=0) # get distances between segments
    m = np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
    s = np.cumsum(m) # cumulative summation of magnitudes
    s/=s[-1] # normalize distances using its total length

    # Find closest index boundaries
    s = np.insert(s,0,0) # prepend with 0 for proper index matching
    i0 = (s.searchsorted(u,side='left')-1).clip(min=0) # Find closest lowest boundary position
    i1 = i0+1 # upper boundary will be the next up

    # Return projection on segments for each query
    if projection:
        return ((p[i1]-p[i0])*((u-s[i0])/(s[i1]-s[i0]))[:,None])+p[i0]

    # Else, modulate parametric samples and and pass back to splev
    mod = (((u-s[i0])/(s[i1]-s[i0]))/steps)+samples[i0]
    return np.array(interpolate.splev(mod,tck)).T 

def airfoilSpline(path, num_points=1000, method='spline',show=False):
    pts = np.genfromtxt(path,delimiter=',',skip_header=0)
    pts = remDupPts(pts)
    if method == 'spline':
        pts2 = splineQuery(pts,np.linspace(0,1,num_points),steps=10000, projection=False)
    elif method == 'linear':
        dists = [0] + [ptDist(pts[i,:], pts[i-1,:]) for i in range(1,pts.shape[0])]
        dists = np.cumsum(dists)
        dists = dists/dists[-1]
        ts = np.linspace(0,1,num_points)
        xs = np.interp(ts, dists, pts[:,0])
        ys = np.interp(ts, dists,pts[:,1])
        pts2 = np.transpose(np.vstack((xs,ys)))
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pts[:,0], pts[:,1], 'o', fillstyle='none', markeredgecolor='k', markersize=4)
        ax.plot(pts2[:,0], pts2[:,1], '-', marker='.', markersize=3, color='r', markerfacecolor='r')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    return pts2

def interpolateUIUCAirfoils(method):
    files = os.listdir('UIUC/csv/')
    if not os.path.isdir('UIUC/airfoil/'):
                os.mkdir('UIUC/airfoil/')
    for f in files[10:]:
        print('Resampling Airfoil: {}'.format(f))
        pts = airfoilSpline('UIUC/csv/'+f,num_points=1000,method=method,show=False)
        np.savetxt('UIUC/airfoil/'+f, pts, delimiter=',')

def lookupUIUCData(fname, re, ncrit):
    has_data = True
    try: 
        pts = np.genfromtxt('UIUC/airfoil/'+fname,delimiter=',',skip_header=0)
        polar = np.genfromtxt('UIUC/polars/uiuc_re{}_ncrit{}/{}'.format(re,ncrit,fname),delimiter=',',skip_header=1)
    except OSError:
        has_data = False

    if has_data:
        #remove nan in polars
        rmidx = []
        for i in range(polar.shape[0]):
            if np.isnan(np.sum(polar[i,:])):
                rmidx.append(i)
            elif np.any(polar[i,1:]> 8):
                rmidx.append(i)
                print(polar[i,:])
        polar = polar[[x for x in range(polar.shape[0]) if x not in rmidx], :]
        
        set_pts = np.zeros((polar.shape[0],pts.shape[0],2))
        set_alpha = polar[:,0]
        set_cl = polar[:,1]
        set_cd = polar[:,2]
        set_cm = polar[:,4]
        # make origin the centroid of the airfoil
        center = np.mean(pts, axis=0)
        center = np.tile(center, [pts.shape[0],1])
        pts = pts - center

        for i in range(polar.shape[0]):
            new_pts = rotatePts(pts, polar[i,0])
            set_pts[i,:,:] = new_pts
        set_pts = np.swapaxes(set_pts, 1,2)
    return {'pts': set_pts, 'cl': set_cl, 'cd': set_cd, 'cm': set_cm, 'a': set_alpha}

def createUIUCDataset(re, ncrit):
    # create dataset
    files = os.listdir('UIUC/airfoil/')

    all_pts = [None for _ in range(len(files))]
    all_alpha = [None for _ in range(len(files))]
    all_cl = [None for _ in range(len(files))]
    all_cd = [None for _ in range(len(files))]
    all_cm = [None for _ in range(len(files))]
    for nf in tqdm(range(len(files))):
        f = files[nf]
        name = f[:-4]
        has_data = True
        try: 
            pts = np.genfromtxt('UIUC/airfoil/'+f,delimiter=',',skip_header=0)
            polar = np.genfromtxt('UIUC/polars/uiuc_re{}_ncrit{}/{}'.format(re,ncrit,f),delimiter=',',skip_header=1)
        except OSError:
            has_data = False

        if has_data:
            #remove nan in polars
            rmidx = []
            for i in range(polar.shape[0]):
                if np.isnan(np.sum(polar[i,:])):
                    rmidx.append(i)
                elif np.any(polar[i,1:]> 8):
                    rmidx.append(i)
                    print(polar[i,:])
            polar = polar[[x for x in range(polar.shape[0]) if x not in rmidx], :]
            
            set_pts = np.zeros((polar.shape[0],pts.shape[0],2))
            set_alpha = polar[:,0]
            set_cl = polar[:,1]
            set_cd = polar[:,2]
            set_cm = polar[:,4]

            # make origin the centroid of the airfoil
            center = np.mean(pts, axis=0)
            center = np.tile(center, [pts.shape[0],1])
            pts = pts - center
            
            for i in range(polar.shape[0]):
                new_pts = rotatePts(pts, polar[i,0])
                set_pts[i,:,:] = new_pts
            all_pts[nf] = set_pts
            all_cl[nf] = set_cl
            all_cd[nf] = set_cd
            all_cm[nf] = set_cm
            all_alpha[nf] = set_alpha

    all_pts = [x for x in all_pts if x is not None]
    all_cl = [x for x in all_cl if x is not None]
    all_cd = [x for x in all_cd if x is not None]
    all_cm = [x for x in all_cm if x is not None]
    all_alpha = [x for x in all_alpha if x is not None]

    all_pts = np.concatenate(all_pts,axis=0)
    all_pts = np.swapaxes(all_pts,1,2)
    all_cl = np.concatenate(all_cl,axis=0)
    all_cd = np.concatenate(all_cd,axis=0)
    all_cm = np.concatenate(all_cm,axis=0)
    all_alpha = np.concatenate(all_alpha,axis=0)
    all_cl = all_cl[:,np.newaxis]
    all_cd = all_cd[:,np.newaxis]
    all_cm = all_cm[:,np.newaxis]
    all_alpha = all_alpha[:,np.newaxis]

    with h5py.File('UIUC/uiuc_dataset_re{}_ncrit{}.h5'.format(re, ncrit), 'w') as f:
        dset = f.create_dataset("geometry", data=all_pts)
        dset = f.create_dataset("cl", data=all_cl)
        dset = f.create_dataset("cd", data=all_cd)
        dset = f.create_dataset("cm", data=all_cm)
        dset = f.create_dataset("a", data=all_alpha)