# @File(label="Select a file") Experiment
from ij import IJ
from loci.formats import ImageReader
from loci.formats import MetadataTools
from ome.units import UNITS
from datetime import datetime
import time

def time_parser():
    """ Iterates through timelapse,                           """
    """ outputs timepoints with corresponding seriesnames.    """
    """ - S. Grødem 2017                                      """
    
    # Get metadata.
    reader = ImageReader()
    omeMeta = MetadataTools.createOMEXMLMetadata()
    reader.setMetadataStore(omeMeta)
    reader.setId(str(Experiment))

	# Extracts number of image series, channel number
    seriesCount = reader.getSeriesCount()
    reader.close()

    # Gets timepoints, in minutes.
    timelist = []
    namelist = []
    
    for timepoint in range (seriesCount):
        times = omeMeta.getImageAcquisitionDate(timepoint)
        timelist.append(times.toString())
        namelist.append(omeMeta.getImageName(timepoint))

    # YY.MM... to minutes.
    timelist =[ time.mktime(time.strptime(times, u'%Y-%m-%dT%H:%M:%S')) for times in timelist ]
    timelist_unsorted =[ (times - timelist[0])/60 for times in timelist ]

    # Sort timepoints.
    timelist, namelist = zip(*sorted(zip(timelist_unsorted, namelist)))
    timelist = [round(float(i), 3) for i in timelist]

    # Output to IJ log
    images = zip(timelist, namelist)
    IJ.log("Series number: " + str(seriesCount))
    IJ.log("*"*15)
    
    for i in range(len(images)):
        IJ.log("Name: " + str(images[i][1]))
        IJ.log("Time: " + str(images[i][0]))
        IJ.log("-"*15)

time_parser()