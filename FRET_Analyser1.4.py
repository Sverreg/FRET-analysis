# @String(label="Experiment Title", description="Set the title of your experiment") Title
# @File(label="Select a file") Experiment
# @File(label="Select Root directory", style="directory") Root
# @Integer(label="Control series: ", description="The number of baseline measurements", value=3) Control_num
from math import *
import os
import collections
from ij.gui import *
from ij.plugin import *
from ij.io import DirectoryChooser
from jarray import array
from ij import IJ, WindowManager, ImagePlus
from ij.process import ImageConverter
from ij.process import ImageProcessor
from ij.gui import GenericDialog
from ij.plugin import ZProjector
from ij.plugin import CompositeConverter
from loci.plugins import BF
from loci.formats import ImageReader
from loci.formats import MetadataTools
from ome.units import UNITS
from loci.common import Region
from loci.plugins.in import ImporterOptions
from trainableSegmentation import WekaSegmentation
from fiji.threshold import Auto_Threshold
from ij.plugin.filter import ThresholdToSelection
from ij.plugin.filter import EDM
from ij.plugin.filter import Binary
from ij.plugin.filter import Analyzer
from ij.plugin.filter import BackgroundSubtracter
from ij.plugin.filter import *
import Watershed_Irregular_Features
from ij.plugin.frame import RoiManager
from ij.gui import Roi
from register_virtual_stack import Register_Virtual_Stack_MT
from ij.measure import ResultsTable
import itertools
from itertools import repeat, chain
import register_virtual_stack.Transform_Virtual_Stack_MT
import json
from datetime import datetime
from ij.gui import Plot
from ij.gui import WaitForUserDialog
from java.awt import Color
from java.awt import Dimension
import time
from java.awt import Font
from ij.plugin import ImageCalculator
from ij.plugin import Duplicator

def FRET_analyser():
	""" Master method and tabulator. """

	# Analysis timer, start.
	startTime = datetime.now()

	# Metadata parser	
	channels, timepoints, timelist, timelist_unsorted, LP, org_size = meta_parser()
		
	# User inputs
	Input_data, Input_data_JSON, Stim_List = user_input()

    # Directory spawner.
	dirs = directorator(Title, str(Root))	
	
	# Projections
	imageprojector(channels, timelist_unsorted, dirs)
		
	# Composite image generator
	Compositor(timepoints, channels, dirs)

	# Background subtracter.
	Backgroundremoval(dirs)
	
	# Composite image aligner.
	Composite_Aligner(channels, dirs)
		
	# Raw image transformer (registration).
	Transformer(channels, dirs)
	
	# Composite image segmentation.
	segmentation = Weka_Segm(dirs)
	

	
	# seFRET/3-channel mode. Calls measurement method, 
	# performs calculations, writes results to txt table,
	# calls for plots, gifs and ratiometric image generation.
	if channels == 3:
		# Measurements and calculations.
		FRET_values = Measurements(channels, timelist, dirs)
		
		cFRET, dFRET, aFRET, Raw_ratio, dtoa, A_Conc, norm_aFRET, norm_dFRET, norm_raw, norm_aFRET_self, norm_dFRET_self, norm_raw_self = three_cube(FRET_values, LP)

		# Tabulator.
		results_table = []
 		table = open(os.path.join(dirs["Tables"], "Resultstable.txt"), "w")

		[[results_table.append([a, b, c, d, e, f])for a, b, c, d, e, f in zip(s1,s2,s3,s4,s5,s6)]
		for s1,s2,s3,s4,s5,s6 in zip(Raw_ratio, cFRET, dFRET, aFRET, FRET_values["Slices"], FRET_values["Time"])]
								
		table.write("\t\t\t".join(map(str,["Raw", "cFRET", "dFRET", "aFRET", "Slice", "Time "])))
		table.write(("_")*128)
		for line in range (len(results_table)):
			table.write("\n")
			table.write("\n")
			table.write("\t\t\t".join(map(str,results_table[line])))

		table.close()

		# Plots ahoy.
		max_Y, min_Y = plots(Raw_ratio, timelist, FRET_values["Cell_num"], "Raw", Stim_List, dirs)
		plots(dFRET, timelist, FRET_values["Cell_num"], "dFRET", Stim_List, dirs)
		plots(aFRET, timelist, FRET_values["Cell_num"], "aFRET", Stim_List, dirs)
		plots(dtoa, timelist, FRET_values["Cell_num"], "Donor to Acceptor ratio", Stim_List, dirs)
		plots(norm_aFRET, timelist, FRET_values["Cell_num"], "Normalized aFRET", Stim_List, dirs)
		plots(norm_dFRET, timelist, FRET_values["Cell_num"], "Normalized dFRET", Stim_List, dirs)

		#plots(norm_raw, timelist, Cell_number, Title, "Normalized raw", Stim_List)
		plots(norm_aFRET_self, timelist, FRET_values["Cell_num"], "Mean norm aFRET self", Stim_List, dirs)
		plots(norm_dFRET_self, timelist, FRET_values["Cell_num"], "Mean norm dFRET self", Stim_List, dirs)

		# Corrects donor concentration, plots concentrations.
		IDD_list = [ [ IDD + CFRET for (IDD, CFRET) in zip(x, y) ] for (x, y) in zip(FRET_values["IDD"], cFRET) ]
		plots(IDD_list, timelist, FRET_values["Cell_num"], "Donor concentration", Stim_List, dirs)
		plots(A_Conc, timelist, FRET_values["Cell_num"], "Acceptor concentration", Stim_List, dirs)

	# Ratiometric/2-channel mode. Same as for 3ch with less calculations.
	elif channels == 2:
		FRET_values = Measurements(channels, timelist, dirs)
		Raw_ratio, Sergei_ratio, norm_raw, norm_Sergei= Ratiometric(FRET_values)

		
		results_table = []
 		table = open(os.path.join(dirs["Tables"], "Resultstable.txt"), "w")

		[[results_table.append([a, b, c, d])for a, b, c, d in zip(s1,s2,s3,s4)]
		for s1,s2,s3,s4 in zip(Raw_ratio, Sergei_ratio, FRET_values["Slices"], FRET_values["Time"])]

		table.write("\t\t".join(map(str,["Raw", "Sergei", "Slice", "Time"])))
		for line in range (len(results_table)):
			table.write("\n")
			table.write("\t\t".join(map(str,results_table[line])))

		table.close()

		plots(Raw_ratio, timelist, FRET_values["Cell_num"], "Raw", Stim_List, dirs)
		plots(Sergei_ratio, timelist, FRET_values["Cell_num"], "Sergei", Stim_List, dirs)
		plots(norm_Sergei, timelist, FRET_values["Cell_num"], "Normalized Sergei", Stim_List, dirs)
		plots(norm_raw, timelist, FRET_values["Cell_num"], "Normalized Raw", Stim_List, dirs)

	# Scale, ROI color coded overlay and gif animation.
	Overlayer(org_size, dirs)

	# Ratiometric image generator.
	#ratiometric(LP, org_size, max_Y, min_Y)

	# Done, prints/logs time used.
   	print ("Finished analysis in: "+str(datetime.now()-startTime))
   	IJ.log("Finished analysis in: "+str(datetime.now()-startTime))
   	

def meta_parser():
    """ Iterates through .lif XML/OME metadata, returns selected values eg. timepoints, channels, series count, laser power.. """

    # Get metadata.
    reader = ImageReader()
    omeMeta = MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(omeMeta)
    reader.setId(str(Experiment))

    # Extracts number of image series, channel number
    seriesCount = reader.getSeriesCount()
    channels = reader.getSizeC()
    #reader.close()

    # Number of images
    imageCount = omeMeta.getImageCount()

    # Image size in pixels AND microns (for scalebar).
    Physical_x = omeMeta.getPixelsPhysicalSizeX(0)
    Pixel_x = omeMeta.getPixelsSizeX(0)
    Physical_x = Physical_x.value()
    Pixel_x = Pixel_x.getNumberValue()

    # Assumes square image (x=y).
    org_size = (Physical_x*Pixel_x)*2

    # Laser power of donor excitation laser.
    if channels == 3:
        LP = omeMeta.getChannelLightSourceSettingsAttenuation(0,0)
        LP = 1 - LP.getNumberValue()
    else:
        LP = 0



    timelist = []
    for timepoint in range (imageCount):
        times = omeMeta.getImageAcquisitionDate(timepoint)
        
        timelist.append(times.toString())
	
	
    # YY.MM... to minutes.
    timelist =[ time.mktime(time.strptime(times, u'%Y-%m-%dT%H:%M:%S')) for times in timelist ]
    timelist_unsorted =[ (times - timelist[0])/60 for times in timelist ]

    timelist = sorted(timelist_unsorted)

    # Prints to log.
    IJ.log("Total # of image series (from BF reader): " + str(seriesCount))
    IJ.log("Total # of image series (from OME metadata): " + str(imageCount))
    IJ.log("Total # of channels (from OME metadata): " + str(channels))
    IJ.log("Laserpower (from OME metadata): " + str(LP))
    return channels, seriesCount, timelist, timelist_unsorted, LP, org_size



def directorator(Title, Root):
    """ Creates all required directories, adds (#) if experiment """
    """ replicates exist, returns dict of directory paths.      """

    if not os.path.exists(Root):
        os.makedirs(Root)

    Exp_root_base = os.path.join(Root, Title)
    Exp_root = Exp_root_base
    
    Replicate = 1
    while os.path.exists(Exp_root):
        Exp_root = Exp_root_base + "(%d)" % (Replicate)
        Replicate += 1
        
    
    os.makedirs(Exp_root)

    subdirs = [
        "Plots", "Tables", "Projections",
        "Projections_C0", "Projections_C1",
        "Projections_C2", "Composites",
        "Composites_Aligned", "Transformations",
        "Aligned_All", "Overlays",
        "Gifs"
    ]
    
    dirs = {}    
    for Dest in subdirs:
        os.makedirs(os.path.join(Exp_root, Dest))
        dirs[Dest] = (os.path.join(Exp_root, Dest))

    return dirs

    
def imageprojector(channels, timelist_unsorted, dirs):
	""" Projects .lif timepoints and saves in a common directory,
	    as well as channel separated directories. """
	
	# Defines in path
	path = str(Experiment)

	# BF Importer
	options = ImporterOptions()
	
	try:
		options.setId(path)
	except Exception(e):
		print str(e)
		
	options.setOpenAllSeries(True)
	options.setSplitTimepoints(True)
	options.setSplitChannels(True)
	imps = BF.openImagePlus(options)

	print "UNSORTED", timelist_unsorted
	
	timelist = [x for item in timelist_unsorted for x in repeat(item, channels)]
	print "SORTED", timelist
	timelist, imps = zip(*sorted(zip(timelist, imps)))

	
	counter_C0 = -1
	counter_C1 = -1
	counter_C2 = -1
	# Opens all images, splits channels, z-projects and saves to disk
	for imp in (imps):
		print str(imp)
		# Projection, Sum Intensity
   		project = ZProjector()
		project.setMethod(ZProjector.SUM_METHOD)
		project.setImage(imp)
		project.doProjection()
		impout = project.getProjection()
		projection = impout.getTitle()

		try:
			# Saves channels to disk, 
			# add more channels here if desired, 
			# remember to define new counters.
			if "C=0" in projection:
				counter_C0 += 1
				IJ.saveAs(impout, "TIFF", os.path.join(dirs["Projections"],
				          "Scan" + str(counter_C0).zfill(3) + "C0"))
                
				IJ.saveAs(impout, "TIFF", os.path.join(dirs["Projections_C0"],
				          "Scan" + str(counter_C0).zfill(3) + "C0"))		
			
			elif "C=1" in projection:
				counter_C1 += 1
				IJ.saveAs(impout, "TIFF", os.path.join(dirs["Projections"],
				          "Scan" + str(counter_C1).zfill(3) + "C1"))
                         
				IJ.saveAs(impout, "TIFF", os.path.join(dirs["Projections_C1"],
				          "Scan" + str(counter_C1).zfill(3) + "C1"))

			elif "C=2" in projection:
				counter_C2 += 1
				IJ.saveAs(impout, "TIFF", os.path.join(dirs["Projections"],
				          "Scan" + str(counter_C2).zfill(3) + "C2"))
                         
				IJ.saveAs(impout, "TIFF", os.path.join(dirs["Projections_C2"],
				          "Scan" + str(counter_C2).zfill(3) + "C2"))
		
		except IOException:
			print "Directory does not exist"
			return

	IJ.log("Images projected and saved to disk")

	
def Composite_Aligner(channels, dirs):
	""" Aligns composite images, saves to directory. """
		
	# Reference image name (must be within source directory)	
	reference_name = "Timepoint000.tif"
		
	# Shrinkage option (False = 0)
	use_shrinking_constraint = 0
		 
	p = Register_Virtual_Stack_MT.Param()
		
	# SIFT parameters:
	p.sift.maxOctaveSize = 1024
	p.sift.fdSize = 12
	p.sift.initialSigma = 1.2
	p.maxEpsilon = 15
	
	p.sift.steps = 12
	
	# The 2ch images have a ton of features,
    # sift feature detection sensitivity is 
    # accordingly reduced.
	if channels == 2:
		p.sift.steps = 8
	
	# 1 = RIGID, 3 = AFFINE
	p.featuresModelIndex = 1
	p.registrationModelIndex = 3
	
	# The "inlier ratio":
	p.minInlierRatio = 0.05
	
	# Opens a dialog to set transformation options, comment out to run in default mode
	#IJ.beep()
	#p.showDialog()	

	# Executes alignment.
	print ("Aligning...")
	
	reference_name = "Timepoint000.tif"
	Register_Virtual_Stack_MT.exec(dirs["Composites"] + os.sep, 
	                               dirs["Composites_Aligned"] + os.sep,
	                               dirs["Transformations"] + os.sep,
	                               reference_name, p, use_shrinking_constraint)
	
	# Close alignment window.
	imp = WindowManager.getCurrentImage()
  	imp.close()

def Transformer(channels, dirs):
	""" Applies transformation matrices from Composite_Aligner to all raw, 32-bit projections. """

	# Executes transformations for each channel.
	t = register_virtual_stack.Transform_Virtual_Stack_MT
	
	t.exec(dirs["Projections_C0"] + os.sep,
	       dirs["Aligned_All"] + os.sep,
	       dirs["Transformations"] + os.sep,
	       True)
           
	imp = WindowManager.getCurrentImage()
  	imp.close()
  	
	t.exec(dirs["Projections_C1"] + os.sep,
	       dirs["Aligned_All"] + os.sep, 
	       dirs["Transformations"] + os.sep, 
	       True)
           
	imp = WindowManager.getCurrentImage()
  	imp.close()
  	
  	if channels == 3:
		t.exec(dirs["Projections_C2"] + os.sep,
		       dirs["Aligned_All"] + os.sep,
		       dirs["Transformations"] + os.sep, 
		       True)
               
		imp = WindowManager.getCurrentImage()
  		imp.close()

	
def Weka_Segm(dirs):
	""" Loads trained classifier and segments cells """ 
	"""	in aligned images according to training.    """
	
	# Define reference image for segmentation (default is timepoint000).
	w_train = os.path.join(dirs["Composites_Aligned"], "Timepoint000.tif")
	trainer = IJ.openImage(w_train)
	weka = WekaSegmentation()
	weka.setTrainingImage(trainer)
	
	# Select classifier model.
	weka.loadClassifier(os.path.join(str(Root), "Classifiers", "Composite_classifier.model"))
     
	weka.applyClassifier(False)
	segmentation = weka.getClassifiedImage()
	segmentation.show()

	# Convert image to 8bit
	ImageConverter(segmentation).convertToRGB()
	ImageConverter(segmentation).convertToGray8()
		
	# Threshold segmentation to soma only.
	hist = segmentation.getProcessor().getHistogram()
	lowth = Auto_Threshold.IJDefault(hist)
	segmentation.getProcessor().threshold(lowth)
	segmentation.getProcessor().setThreshold(0, 0, ImageProcessor.NO_LUT_UPDATE)
	segmentation.getProcessor().invert()
	segmentation.show()
	
	# Run Watershed Irregular Features plugin, with parameters.
	IJ.run(segmentation, "Watershed Irregular Features",
	      "erosion=20 convexity_treshold=0 separator_size=0-Infinity")

	# Make selection and add to RoiManager.	
	RoiManager()
	rm = RoiManager.getInstance()
	rm.runCommand("reset")
	roi = ThresholdToSelection.run(segmentation)
	segmentation.setRoi(roi)
	rm.addRoi(roi)
	rm.runCommand("Split")


def Measurements(channels, timelist, dirs):
	""" Takes measurements of weka selected ROIs in a generated aligned image stack. """
	
   	# Set desired measurements. 
	an = Analyzer()
	an.setMeasurements(an.AREA + an.MEAN + an.MIN_MAX + an.SLICE)

	# Opens raw-projections as stack.
	test = IJ.run("Image Sequence...",
	              "open=" + dirs["Aligned_All"]
	              + " number=400 starting=1 increment=1 scale=400 file=.tif sort")

	# Calls roimanager.
	rm = RoiManager.getInstance()	
	total_rois = rm.getCount()

	# Deletes artefact ROIs (too large or too small). 
	imp = WindowManager.getCurrentImage()
	for roi in reversed(range(total_rois)):
		rm.select(roi)
		size = imp.getStatistics().area		
		if size < 200:
			rm.select(roi)
			rm.runCommand('Delete')
		elif size > 2000:
			rm.select(roi)
			rm.runCommand('Delete')
		else:
			rm.runCommand("Deselect")

	# Confirm that ROI selection is Ok (comment out for headless run).
	WaitForUserDialog("ROI check", "Control ROI selection, then click OK").show() 
	
	# Measure each ROI for each channel.
	imp = WindowManager.getCurrentImage()
	rm.runCommand("Select All")	
	rm.runCommand("multi-measure measure_all")		
	
	# Close.
	imp = WindowManager.getCurrentImage()
	imp.close()

	# Get measurement results.
	rt = ResultsTable.getResultsTable()
	Area = rt.getColumn(0)
	Mean = rt.getColumn(1)
	Slice = rt.getColumn(27)
	
	# Removes (and counts) artefact ROIs (redundant)
	# Area indices without outliers
	Area_indices = [index for (index, value) in enumerate(Area, start=0)
	                if value > 200 and value < 2000]
	
	# Mean without outliers from area (redundant)
	Filtered_mean = [Mean[index] for index in Area_indices]
	Filtered_slice = [Slice[index] for index in Area_indices]

	# Number of cell selections.
	Cell_number = Filtered_slice.count(1.0)
	rm = RoiManager.getInstance()
	print "Number of selected cells: ", Cell_number
	print "Total number of selections: ", rm.getCount()

	Cells = [ Filtered_mean [x : x + Cell_number]
	          for x in xrange (0, len(Filtered_mean), Cell_number) ]
              	
	Cells_indices = [ index for (index, value) in enumerate(Cells) ]
	
	time = [ x for item in timelist for x in repeat(item, Cell_number) ]
	time = [ time [x : x + Cell_number] for x in xrange (0, len(time), Cell_number) ]
	
	Slices = [ Filtered_slice [x : x + Cell_number]
	           for x in xrange (0, len(Filtered_slice), Cell_number) ]
	
	# Lists IDD, IDA + IAA if 3ch.
	if channels == 3:
		IDD_list = [ Cells [index] for index in Cells_indices [0::int(channels)] ]
		IDA_list = [ Cells [index] for index in Cells_indices [1::int(channels)] ]	
		IAA_list = [ Cells [index] for index in Cells_indices [2::int(channels)] ]
		
		FRET_values = {"IDD" : IDD_list, "IDA" : IDA_list, "IAA" : IAA_list,
				  	   "Cell_num" : Cell_number, "Slices" : Slices,
				       "Time" : time
				       }
	
	
	elif channels == 2:
		IDD_list = [ Cells [index] for index in Cells_indices [0::int(channels)] ]
		IDA_list = [ Cells [index] for index in Cells_indices [1::int(channels)] ]
		
		FRET_values = {"IDD": IDD_list, "IDA" : IDA_list,
					   "Cell_num" : Cell_number, "Slices" : Slices,
					   "Time" : time
					   }

	return FRET_values
	

def three_cube(FRET_values, LP):
    """ Performs calculations on nested list data from three channels, 
            returns calculations as nested lists. """
    
    # Direct excitation of acceptor, AER, and 
    # donor emission bleedthrough, DER, cofficients.
    # (Determined experimentally.)
    AER = ((8.8764*(LP**2)) + (1.8853*LP)) - 0.1035 
    DER = 0.1586

    IDD_list, IDA_list, IAA_list = FRET_values["IDD"], FRET_values["IDA"], FRET_values["IAA"]
    
    Cell_number = FRET_values["Cell_num"]
    
    # Calculations.
    Raw_ratio = [ [ IDA / IDD for (IDA, IDD) in zip(x, y) ] for (x, y) in zip(IDA_list, IDD_list) ]
    
    cFRET = [ [ IDA - (AER*IAA) - (DER*IDD) 
                for (IDA, IAA, IDD) in zip(x, y, z) ] 
                for (x, y, z) in zip(IDA_list, IAA_list, IDD_list) ]
    
    dFRET = [ [ (IDA - (AER*IAA) - (DER*IDD)) / (IDD + CFRET) 
                for (IDA, IDD, IAA, CFRET) in zip(x, y, z, c) ] 
                for (x, y, z, c) in zip(IDA_list, IDD_list, IAA_list, cFRET) ]
    
    aFRET = [ [ ((IDA - (AER*IAA) - (DER*IDD)) / (IAA*AER))/4.66 
                for (IDA, IDD, IAA) in zip(x, y, z) ] 
                for (x, y, z) in zip(IDA_list, IDD_list, IAA_list) ]    
    
    dtoa = [ [ ((IDD + CFRET)/4.66) / (IAA*AER) 
                for (IDD, CFRET, IAA) in zip(x, y, z) ] 
                for (x, y, z) in zip(IDD_list, cFRET, IAA_list) ]
    
    # Flattens nested lists for normalization function.
    flat_aFRET = [item for sublist in aFRET for item in sublist]
    flat_dFRET = [item for sublist in dFRET for item in sublist]
    flat_raw = [item for sublist in Raw_ratio for item in sublist]

    aFRET_base = flat_aFRET [0:Cell_number]
    aFRET_base_mean = standard_deviation(aFRET_base)
    print aFRET_base_mean
    print aFRET_base
    dFRET_base = flat_dFRET [0:Cell_number]
    dFRET_base_mean = standard_deviation(dFRET_base)
    print dFRET_base_mean
    print dFRET_base
    
    """ Calculates baseline-normalized values. """
    norm_aFRET, norm_aFRET_self = [], []
    norm_dFRET, norm_dFRET_self = [], []
    norm_raw, norm_raw_self = [], []
    baseline = Cell_number * Control_num
    
    for cell in range(Cell_number):
                    # Divides value 0->max by baseline point 0-Cell_number for each cell.
            norm_aFRET.append([ (flat_aFRET[v]) / ((sum(flat_aFRET[cell:baseline:Cell_number]))
                                /len(flat_aFRET[cell:baseline:Cell_number])) 
                                for v in range(cell, len(flat_aFRET), Cell_number)])
            
            norm_dFRET.append([ (flat_dFRET[v]) / ((sum(flat_dFRET[cell:baseline:Cell_number]))
                                /len(flat_dFRET[cell:baseline:Cell_number]))
                                for v in range(cell, len(flat_dFRET), Cell_number)])
                    
            norm_raw.append([ (flat_raw[v]) / ((sum(flat_raw[cell:baseline:Cell_number]))
                              /len(flat_raw[cell:baseline:Cell_number])) 
                              for v in range(cell, len(flat_raw), Cell_number)])
            
            norm_aFRET_self.append([ (flat_aFRET[v]) / (flat_aFRET[cell])
                                     for v in range(cell, len(flat_aFRET), Cell_number)])
            
            norm_dFRET_self.append([ (flat_dFRET[v]) / (flat_dFRET[cell]) 
                                     for v in range(cell, len(flat_dFRET), Cell_number)])   
            
            norm_raw_self.append([ (flat_raw[v]) / (flat_raw[cell])
                                    for v in range(cell, len(flat_raw), Cell_number)])
            
    # Zips [[cell1],[cell2]] to [[time1],[time2]].
    norm_aFRET = list(chain.from_iterable(zip(*norm_aFRET)))
    norm_dFRET = list(chain.from_iterable(zip(*norm_dFRET)))
    norm_raw = list(chain.from_iterable(zip(*norm_raw)))
    norm_aFRET_self = list(chain.from_iterable(zip(*norm_aFRET_self)))
    norm_dFRET_self = list(chain.from_iterable(zip(*norm_dFRET_self)))
    norm_raw_self = list(chain.from_iterable(zip(*norm_raw_self)))  
    
    # Rounds final outputs.
    Raw_ratio = [ [round(float(i), 3) for i in nested] for nested in Raw_ratio ]
    dFRET = [ [round(float(i), 3) for i in nested] for nested in dFRET ]
    aFRET = [ [round(float(i), 3) for i in nested] for nested in aFRET ]
    cFRET = [ [int(i) for i in nested] for nested in cFRET ]
    dtoa = [ [round(float(i), 3) for i in nested] for nested in dtoa ]
    norm_aFRET = [round(float(i), 5) for i in norm_aFRET]
    norm_dFRET = [round(float(i), 5) for i in norm_dFRET]
    norm_raw = [round(float(i), 5) for i in norm_raw]
    
    A_Conc = [ [ (IAA * AER) for (IAA) in x ] for x in IAA_list ]

    return cFRET, dFRET, aFRET, Raw_ratio, dtoa, A_Conc, norm_aFRET, norm_dFRET, norm_raw, norm_aFRET_self, norm_dFRET_self, norm_raw_self
    
    


def Ratiometric(FRET_values):
    """ Performs calculations on nested list data from two channels, 
        returns calculations as nested lists. """

    IDD_list, IDA_list = FRET_values["IDD"], FRET_values["IDA"]
    Cell_number = FRET_values["Cell_num"]
    # Calculations
    Raw_ratio = [ [ IDA / IDD for (IDA, IDD) in zip(x, y) ] 
                    for (x, y) in zip(IDA_list, IDD_list) ]
                 
    Sergei_ratio = [ [ (IDA / IDD) - 0.33 for (IDA, IDD) in zip(x, y) ] 
                        for (x, y) in zip(IDA_list, IDD_list) ]
    
    baseline = Cell_number * Control_num
    norm_raw, norm_Sergei = [], []

    flat_raw = [item for sublist in Raw_ratio for item in sublist]

    flat_Sergei = [item for sublist in Sergei_ratio for item in sublist]
    
    for cell in range(Cell_number):
        norm_raw.append([ (flat_raw[v]) / ((sum(flat_raw[cell:baseline:Cell_number]))
                          /len(flat_raw[cell:baseline:Cell_number])) 
                          for v in range(cell, len(flat_raw), Cell_number)])
                          
        norm_Sergei.append([ (flat_Sergei[v]) / ((sum(flat_Sergei[cell:baseline:Cell_number]))
                             /len(flat_Sergei[cell:baseline:Cell_number]))
                             for v in range(cell, len(flat_Sergei), Cell_number)])
    
    norm_raw = list(chain.from_iterable(zip(*norm_raw)))
    norm_Sergei = list(chain.from_iterable(zip(*norm_Sergei)))
    # Round.
    Raw_ratio = [ [round(float(i), 3) for i in nested] for nested in Raw_ratio ]
    Sergei_ratio = [ [round(float(i), 3) for i in nested] for nested in Sergei_ratio ]
    norm_raw = [round(float(i), 5) for i in norm_raw]
    norm_Sergei = [round(float(i), 5) for i in norm_Sergei]
        
    return Raw_ratio, Sergei_ratio, norm_raw, norm_Sergei


		
def Compositor(timepoints, channels, dirs):
	""" Creates composite images of all channels for each timepoint. """
	 	
  	# Creates composite stack from HDD images.
 	for time in range (1, (timepoints*channels), channels):
  		
  		stack = IJ.run(
  		    "Image Sequence...", "open="
  		    + dirs["Projections"] + " number=" + str(channels)
  		    + " starting=" + str(time) + " increment=1 scale=400 file=.tif sort")
        
  		comp = IJ.run("Make Composite", "display=Composite")
  		comp = IJ.run("Stack to RGB", comp)
       
  		IJ.saveAs(comp, "Tiff", os.path.join(dirs["Composites"],
  		         "Timepoint"+str(time/channels).zfill(3)))

		# Close windows.
  		for i in range (2):
  			try:
  				imp = WindowManager.getCurrentImage()
  				imp.close()
  			except:
  				pass


def Backgroundremoval(dirs):
	""" Runs rolling ball background subtraction on all channels. """
	
	# Processes channel 1.. etc
	for root, directories, filenames in os.walk(dirs["Projections_C0"]):
		for filename in filenames:
      		# Check for file extension
			if filename.endswith(".tif"):		
				process(dirs["Projections_C0"], root, filename)

	for root, directories, filenames in os.walk(dirs["Projections_C1"]):
		for filename in filenames:
      		# Check for file extension
			if filename.endswith(".tif"):		
				process(dirs["Projections_C1"], root, filename)

	for root, directories, filenames in os.walk(dirs["Projections_C2"]):
		for filename in filenames:
      		# Check for file extension
			if filename.endswith(".tif"):		
				process(dirs["Projections_C2"], root, filename)

 		
def process(Destination_Directory, Current_Directory, filename):
	""" Rolling ball method. """
    
	print "Processing:"   
  	# Opening the image
  	print "Open image file", filename
	
	imp = IJ.openImage(os.path.join(Current_Directory, filename))
	ip = imp.getProcessor()
	
	# Parameters: Image processor, Rolling Ball Radius, Create background, 
	#             light background, use parabaloid, do pre smoothing (3x3), 
	# 			  correct corners
	b = BackgroundSubtracter()	
	b.rollingBallBackground(ip, 50, False, False, True, False, False)	

	print "Saving to", Destination_Directory	
	IJ.saveAs(imp, "Tiff", os.path.join(Destination_Directory, filename))	
	imp.close()


def Overlayer(org_size, dirs):
	""" Overlays ROIs with appropriate color,
	    saves to .tif and animates aligned images to .gif """
    
    # Get colors.
	Colors, Colors_old = colorlist()

    # Get ROImanager.
	rm = RoiManager().getInstance()
	rois = rm.getCount()
	
	# Overlays ROI on aligned images, converts to 8-bit (for gif).
	for root, directories, filenames in os.walk(dirs["Composites_Aligned"]):
		for filename in filenames:
			imp = IJ.openImage(os.path.join(root, filename))
			converter = ImageConverter(imp)
			converter.setDoScaling(True)
			converter.convertToGray8()

			# Lookup table and local contrast enhancement for vizualisation.
			IJ.run(imp, "Rainbow RGB", "")
			IJ.run(imp, "Enhance Local Contrast (CLAHE)", 
			       "blocksize=127 histogram=256 maximum=3 mask=*None*")

			
			for roi in range(rois):
				roi_obj = rm.getRoi(roi)
				roi_obj.setStrokeWidth(2)		
				if roi < 19:
					roi_obj.setStrokeColor(Color(*Colors[roi][0:3]))
				else:
					roi_obj.setStrokeColor(eval(Colors_old[roi]))
			
			
			rm.moveRoisToOverlay(imp)

			IJ.saveAs(imp, "Tiff", os.path.join(dirs["Overlays"], filename))
			
	# Opens overlaid images, saves as tiff stack.
	overlay_stack = IJ.run("Image Sequence...", "open="+dirs["Overlays"]+
					       " number=300 starting=0 increment=1 scale=300 file=.tif sort")
	
	# Takes care of spaces in titles. 
	tiftitle = Title.replace(" ", "_")
	tiftitle = tiftitle.replace(".", "_")
	
	# Gets dimensions for scalebar.
	imp = WindowManager.getImage("Overlays")
	dimensions = imp.getDimensions()
	size = dimensions[0] + dimensions[1]
	microns = org_size / size

	# Sets scale and writes scale-bar, flattens overlays. 
	IJ.run(imp, "Set Scale...", "distance=1 known="
	       +str(microns)+" pixel=1 unit=micron")
           
	IJ.run(imp, "Scale Bar...", 
	    "width=10 height=4 font=14 color=Yellow background=None location=[Lower Right] bold overlay")
    
	IJ.run(imp, "Flatten", "stack")
	IJ.saveAs(imp, "Tiff", os.path.join(dirs["Gifs"], tiftitle))
	
	# Animates tiff stack from directoy. 
	for root, directories, filenames in os.walk(dirs["Gifs"]):
		for filename in filenames:		
			if tiftitle in filename and filename.endswith(".tif"):
				# set=xx parameter controls gif speed.
				# for additional parameters run with macro recorder.
				try:
					print "Animating gif..."
					imp = WindowManager.getImage(tiftitle + ".tif")
					gif = IJ.run("Animated Gif ... ", 
					             "set=200 number=0 filename="
					             + os.path.join(dirs["Gifs"], tiftitle + ".gif"))
				
				except Exception, e:
					print str(e)
				
				print "gif animated."

	# Close.
	imp = WindowManager.getCurrentImage()
	imp.close()

	
def user_input():
	""" Takes user input on experiment parameters and stimulation applications... """
	
	User_Input_List = []
	Stim_List = []
	
	# Takes title, description and number of stims from user.
	gd = GenericDialog("Experiment Parameters:")
	gd.addStringField("Experiment Description:", "")
  	gd.addNumericField("Number of stimulation points:", 1, 1)

  	# Gets values from dialogs.
  	gd.showDialog()
  	Description = gd.getNextString()
  	Nr_Stim = int(gd.getNextNumber())

	User_Input_List.append([Title, Description, Nr_Stim])

	# Creates dialog boxes proportionally to number of stimulations, type and duration. 
  	if Nr_Stim >= 1:
  		gd = GenericDialog("Stimulation applications")
  		for stim in range(0, Nr_Stim, 1):
  			gd.addStringField("Stimulation type "+str(stim+1)+":", "cLTP")
  			gd.addNumericField("Stimulation start:", stim*2, 2)
  			gd.addNumericField("Stimulation end:", (stim+1)*2, 2)
  		
  		gd.showDialog()

  	# Lists the different stimulations.
	for stim in range(0, Nr_Stim, 1):
		Type_Stim = gd.getNextString()
  		Start_Stim = gd.getNextNumber()
  		End_Stim = gd.getNextNumber()
  		Stim_List.append([Type_Stim, float(Start_Stim), float(End_Stim)])

	User_Input_List.extend(Stim_List)
	
	# Creates a dictionary of all user inputs.
	User_Input_Dict = {'Parameters': User_Input_List[0]}
	for stim in range(1, Nr_Stim+1, 1):
		User_Input_Dict['Stimulation '+str(stim)] = User_Input_List[stim]

	# Dumps dict to JSON.	
	User_Input_Dict_JSON = (json.dumps(User_Input_Dict))	
	return User_Input_Dict, User_Input_Dict_JSON, Stim_List


def plots(values, timelist, Cell_number, value_type, Stim_List, dirs):
	""" Plots all calculated values, saves plots to generated directory, returns plot scale. """

	Mean_plot = 0
	# Flatten nested lists (normalized lists are not nested).
	if value_type == "Mean norm aFRET self":	
		values_concat = [ values[i:i+Cell_number] for i in range(0, (len(values)), Cell_number) ]
		Mean_sd = [ standard_deviation(values_concat[i]) for i in range(len(values_concat)) ]
		Mean_sd = [item for sublist in Mean_sd for item in sublist]
		Mean_plot = 1
	elif value_type == "Mean norm dFRET self":
		values_concat = [ values[i:i+Cell_number] for i in range(0, (len(values)), Cell_number) ]
		Mean_sd = [ standard_deviation(values_concat[i]) for i in range(len(values_concat)) ]
		Mean_sd = [item for sublist in Mean_sd for item in sublist]
		Mean_plot = 1

	else:
		if "Normalized" not in value_type:
			values = [item for sublist in values for item in sublist]

	#Repeats list items x cell_number (match timepoints with # of cells).
	timelist = [x for item in timelist for x in repeat(item, Cell_number)]

	# Scaling of plots.
	max_Y = 1
	if max(values) > 3:
		if not isinstance(values[0], list):
			max_Y = max(values)*1.3
	elif max(values) > 2.5:
		max_Y = 3.3
	elif max(values) > 2:
		max_Y = 2.7
	elif max(values) > 1.5:
		max_Y = 2.2
	elif max(values) > 1.3:
		max_Y = 1.7
	elif max(values) > 1:
		max_Y = 1.4


	min_Y = 0
	if min(values) > 2:
		min_Y = min(values)*0.8
	elif min(values) > 1.5:
	    min_Y = 1.5
	elif min(values) > 1:
	    min_Y = 1
	elif min(values) > 0.5:	
	    min_Y = 0.2
			
	elif min(values) < -0.5:
		min_Y = min(values)*1.3
	elif min(values) < -0.2:
	    min_Y = -0.3
	elif min(values) < -0.1:
	    min_Y = -0.15
	elif min(values) < -0.08:
	    min_Y = -0.1
	elif min(values) < -0.05:
	    min_Y = -0.08
	elif min(values) < -0.01:
	    min_Y = -0.06

	# Scaling of normalized plots..
	if "Normalized" in value_type:
		max_Y, min_Y = 1.75, 0.4

	if value_type == "dFRET":
		max_Y = 0.65
		min_y = 0.0
	elif value_type =="aFRET":
		max_Y = 0.65
		min_Y = 0.0

	# Call plot, set scale.
	plot = Plot(Title, "Time (minutes)", value_type)
	plot.setLimits(min(timelist), max(timelist), min_Y, max_Y)
	# Retrieve colors.
	Colors, Colors_old = colorlist()

	# Set colors, plot points.
	if Mean_plot == 0:
	    for i in range(Cell_number):
		    if i < 19:
			    plot.setColor(Color(*Colors[i][0:3]))
		    elif i >= 19:
			    plot.setColor(eval(Colors_old[i]))
			    print "Out of fancy colors, using java.awt.color defaults"
		    elif i > 28:
			    print "29 color limit exceeded"
			    return
			    
	            plot.setLineWidth(1.5)
	            plot.addPoints(timelist[i :: Cell_number], values[i :: Cell_number], Plot.LINE)
	            plot.setLineWidth(1)
	
	            # Comment in to define color + fillcolor for circles.
	            plot.setColor(Color(*Colors[i][0:3]), Color(*Colors[i][0:3]))
	            plot.addPoints(timelist[i :: Cell_number], values[i :: Cell_number], Plot.CIRCLE)
	else:
		min_Y, max_Y = 0.6, 1.6
		plot.setLimits(min(timelist), max(timelist), min_Y, max_Y)
		plot.setColor("Color.BLACK")
		plot.setLineWidth(1.5)
		plot.addPoints(timelist[0 :: Cell_number], Mean_sd[0::2], Plot.LINE)
		plot.setLineWidth(1)
		plot.setColor("Color.BLACK", "Color.BLACK")
		plot.addPoints(timelist[0 :: Cell_number], Mean_sd[0::2], Plot.CIRCLE)
		plot.setColor(Color(*Colors[6][0:3]))
		plot.addErrorBars(Mean_sd[1::2])

	# Get's stim name from input.
	text = [ sublist[i] for sublist in Stim_List for i in range(len(Stim_List)) ]
	Stim_List = [ sublist[1:] for sublist in Stim_List ]

	# Plot stimulation markers. 
	plot.setLineWidth(2)
	for sublist in Stim_List:
	    plot.setColor("Color.GRAY")
	    plot.drawLine(sublist[0], min_Y+((max_Y-min_Y) * 0.82), sublist[1], min_Y+((max_Y-min_Y) * 0.82))
	    plot.drawDottedLine(sublist[0], min_Y+((max_Y-min_Y) * 0.82), sublist[0], -1, 4)
	    plot.drawDottedLine(sublist[1], min_Y+((max_Y-min_Y) * 0.82), sublist[1], -1, 4)
	    plot.setFont(Font.BOLD, 16)
	    plot.addText(text[0], sublist[0], min_Y+((max_Y-min_Y) * 0.82))

    
	if "concentration" not in value_type:
		testfile = open(os.path.join(dirs["Tables"], value_type + ".txt"), "w")
		data = plot.getResultsTable()
		headings = data.getHeadings()
		datadict = {}
		for heading in headings:
			index = data.getColumnIndex(heading)
			column = { heading : data.getColumn(index) }
			datadict.update(column)
		for key, value in datadict.iteritems():
			testfile.write("\n" + key + "\n" + "\t".join([str(round(x, 4)) for x in value]))

		testfile.close()
	
	# Generate High-res plot with anti-aliasing (Scale x 1). 
	plot = plot.makeHighResolution(Title, 1, True, True)	
	#PlotWindow.noGridLines = True

	# Save plot with appropriate title.
	IJ.saveAs(plot, "PNG", os.path.join(dirs["Plots"], str(Title)+str(value_type)))

	# (For ratiometric image-generator)
	return max_Y, min_Y

	
def ratiometric(LP, org_size, max_Y, min_Y):
	""" Generates ratiometric images of dFRET/aFRET for each timepoint. """
	
	source_directory = "D:\\Image_Processing\\Virtualstacks\\Lif_Stack_Split_aligned_all\\"
	destination_directory = "D:\\Image_Processing\\Virtualstacks\\Ratiometrics\\"
	destination_directory2 = "D:\\Image_Processing\\Virtualstacks\\Ratiometrics_withROI\\"

	# As in nested_calculator. 
	AER = ((8.8764*(LP**2)) + (1.8853*LP)) - 0.1035 
	DER = 0.1586

	# File list of every image in every channel.
	channels = []
	for root, directory, filenames in os.walk(source_directory):
		for filename in filenames:
			channels.extend(filename)

	# Channel-separator.
	IDD, IDA, IAA = channels[0::3], channels[1::3], channels[2::3]

	# Create combined ROI.
	rm = RoiManager.getInstance()
	rm.runCommand("Select All")
	rm.runCommand("Combine")
	rm.runCommand("Add")
	roi = rm.getRoi(rm.getCount())
	
	for timepoint in range(len(IDD)):
		# Opens each image, clears outside ROIs.
		IDD_base = IJ.openImage(os.path.join(root, IDD[timepoint]))
		IDD_base_ip = IDD_base.getProcessor()
		IDD_base_ip.fillOutside(roi)
		
		IDA_base = IJ.openImage(os.path.join(root, IDA[timepoint]))
		IDA_base_ip = IDA_base.getProcessor()
		IDA_base_ip.fillOutside(roi)
		
		IAA_base = IJ.openImage(os.path.join(root, IAA[timepoint]))
		IAA_base_ip = IAA_base.getProcessor()
		IAA_base_ip.fillOutside(roi)

		# Duplicates IDD, IAA for subtraction of cross-talk.
		IDD_subtracted = Duplicator().run(IDD_base)
		IAA_subtracted = Duplicator().run(IAA_base)

		IJ.run(IDD_subtracted, "Multiply...", "value="+str(DER))
		IJ.run(IDA_subtracted, "Multiply...", "value="+str(AER))

		# Subtracts cross-talk from IDA.
		cFRET1 = ImageCalculator().run("Subtract create 32-bit", IDA, IDD_subtracted)
		cFRET = ImageCalculator().run("Subtract create 32-bit", cFRET1, IAA_subtracted)

		# Corrected donor concentration.
		donor_c = ImageCalculator().run("Add create 32-bit", IDD_base, cFRET)

		# dFRET.
		dFRET = ImageCalculator().run("Divide create 32-bit", cFRET, donorc)
		dFRET.setTitle("dFRET")
		IJ.run(dFRET, "Rainbow RGB", "")

		# aFRET.
		aFRET = ImageCalculator().run("Divide create 32-bit", cFRET, IAA_base)
		IJ.run(aFRET, "Divide...", "value=4.66")
		aFRET.setTitle("aFRET")
		IJ.run(aFRET, "Rainbow RGB", "")

		# Scalebar.
		dimensions = dFRET.getDimensions()
		size = dimensions[0] + dimensions[1]
		microns = org_size / size
		FRET = [aFRET, dFRET]
		for i in range(2):
			IJ.run(FRET[i], "Set Scale...", "distance=1 known="+str(microns)+" pixel=1 unit=micron")
			IJ.run(FRET[i], "Scale Bar...", "width=10 height=4 font=14 color=Yellow background=None location=[Lower Right] bold overlay")
			#FRET[i].getProcessor().setMinAndMax(0, 1)
			IJ.run(FRET[i], "Calibration Bar...", "location=[Upper Right] fill=None label=White number=3 decimal=2 font=12 zoom=1.5 bold overlay")
			IJ.run(FRET[i], "Flatten", "stack")
		# Save.
		IJ.saveAs(aFRET, "Tiff", os.path.join(destination_directory, IDD[i][0:7]+"aFRET"))
		IJ.saveAs(dFRET, "Tiff", os.path.join(destination_directory, IDD[i][0:7]+"dFRET"))
				
	
def colorlist():
	""" Color library. """
	
	# Kelly's Colors:
        # Vivid Yellow, Strong Purple, Vivid Orange, 
        # Very Light Blue, Vivid Red, Grayish Yellow, 
        # Medium Gray, Vivid Green, Strong Purplish Pink,
        # Strong Blue, Strong Yellowish Pink, Strong Violet,
        # Vivid Orange Yellow, Strong Purplish Red, Vivid Greenish Yellow,
        # Strong Reddish Brown, Vivid Yellowish Green, Deep Yellowish Brown,
        # Vivid Reddish Orange, Dark Olive Green	
	Colors = [[255, 179, 0], [128, 62, 117], [255, 104, 0],
    		 [166, 189, 215], [193, 0, 32], [206, 162, 98],
    		 [129, 112, 102], [0, 125, 52], [246, 118, 142],
    		 [0, 83, 138], [255, 122, 92], [83, 55, 122],
    		 [255, 142, 0], [179, 40, 81], [244, 200, 0],
    		 [127, 24, 13], [147, 170, 0], [89, 51, 21],
    		 [241, 58, 19], [35, 44, 22]]

	#java.awt.colors defaults.
	Colors_old = ["Color.RED", "Color.GREEN", "Color.CYAN",
	"Color.PINK", "Color.ORANGE", "Color.BLUE",
	"Color.MAGENTA", "Color.GRAY", "Color.darkGray"] 

	return Colors, Colors_old

	
def standard_deviation(values):
    count = len(values)
    mean = sum(values) / count
    differences = [ x - mean for x in values ]
    sq_differences = [ d ** 2 for d in differences ]
    ssd = sum(sq_differences)
    if count > 1:
        variance = ssd / (count - 1)
    else:
        variance = ssd / count

    sd = sqrt(variance)
    mean_sd_list = [mean, sd]
    return mean_sd_list


FRET_analyser()
