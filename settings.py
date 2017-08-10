# @File(label="Select Root directory", style="directory") Root
import itertools
from ij import IJ
from ij.gui import GenericDialog
from java.awt import Panel
from java.util import Vector
import ConfigParser
import os

def settings():
    """ Settings """

    # Registration parameters dialog.
    gd = GenericDialog("Advanced Settings")
    gd.addMessage("REGISTRATION PARAMETERS") 
    gd.addNumericField("Steps per scale octave: ", 8, 0, 7, "")
    gd.addNumericField("Max Octave Size: ", 1024, 0, 7, "")
    gd.addNumericField("Feature Descriptor Size: ", 12, 0, 7, "")
    gd.addNumericField("Initial Sigma: ", 1.2, 2, 7, "")  
    gd.addNumericField("Max Epsilon: ", 15, 0, 7, "")
    gd.addCheckbox("  Use Shrinkage Constraint", False)
    
    # Background removal parameters dialog.
    gd.addPanel(Panel())
    gd.addMessage("BACKGROUND REMOVAL")
    gd.addChoice("Subtraction method:", ["Rolling Ball", "Manual selection"], "Rolling Ball")
    gd.addNumericField("Rolling ball size: ", 50, 0, 7, "")
    gd.addCheckbox("  Create Background", False)
    gd.addCheckbox("  Light Background", False)
    gd.addCheckbox("  Use Parabaloid", False)
    gd.addCheckbox("  Do Pre-smoothing", False)
    gd.addCheckbox("  Correct Corners", False)

    # Measumrent parameters dialog.
    gd.addPanel(Panel())
    gd.addMessage("MEASUREMENT PARAMETERS")
    gd.addNumericField("Max Cell Area", 2200, 0, 7, "px")
    gd.addNumericField("Min Cell Area", 200, 0, 7, "px")

    # Plot parameters dialog.
    gd.addPanel(Panel())
    gd.addMessage("PLOT PARAMETERS")
    gd.addNumericField("Max y, d and aFRET", 0.65, 2, 7, "")
    gd.addNumericField("Min y, d and aFRET", 0, 2, 7, "")
    gd.addNumericField("Max y, norm. d and aFRET", 1.65, 2, 7, "")
    gd.addNumericField("Min y, norm. d and aFRET", 0.5, 2, 7, "")
    
    # Set location of dialog on screen.
    #gd.setLocation(0,1000)
    
    gd.showDialog()

    if gd.wasCanceled() is True:
        return False

    parameters = {"Steps" : gd.getNextNumber(), 
                 "Max_oct" : gd.getNextNumber(),
                 "FD_size" : gd.getNextNumber(),
                 "Sigma" : gd.getNextNumber(),
                 "Max_eps" : gd.getNextNumber(),
                 "Shrinkage" : gd.getNextBoolean(),
                 "b_sub" : gd.getNextChoiceIndex(),
                 "ballsize" : gd.getNextNumber(),
                 "Create_b" : gd.getNextBoolean(),
                 "Light_b" : gd.getNextBoolean(),
                 "Parab" : gd.getNextBoolean(),
                 "smooth" : gd.getNextBoolean(),
                 "Corners" : gd.getNextBoolean(),
                 "Cell_max" : gd.getNextNumber(),
                 "Cell_min" : gd.getNextNumber(),
                 "p_max" : gd.getNextNumber(),
                 "p_min" : gd.getNextNumber(),
                 "p_max_n" : gd.getNextNumber(),
                 "p_min_n" : gd.getNextNumber()
                 }

    config_write(parameters)   


def config_write(parameters):
    config = ConfigParser.RawConfigParser()
    config.add_section("Parameters")

    
    for key, value in parameters.iteritems():
        config.set("Parameters", str(key), str(value))

    with open(os.path.join(str(Root), "example.cfg"), "wb") as configfile:
        config.write(configfile)

    return parameters


def config_read():
    config = ConfigParser.RawConfigParser()
    con_path = os.path.join(str(Root), "example.cfg")
    
    if os.path.exists(con_path):
        config.read(con_path)
        steps = config.getfloat("Parameters", "Steps")
        """
        try:
            an_int = config.getint("Section2", "an_int")
        except ConfigParser.NoSectionError:
            print "No such section exists..."
            pass
        
        """
        print steps
    #return parameters

settings()

config_read()



