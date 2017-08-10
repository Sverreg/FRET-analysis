import ConfigParser
import itertools
import os
def config_write():
    config = ConfigParser.RawConfigParser()
    config.add_section('Section1')
    config.set('Section1', 'an_int', 12)

    with open('example2.cfg', 'wb') as configfile:
        config.write(configfile)

    config.read('example.cfg')

def config_read():
    config = ConfigParser.RawConfigParser()

    
    if os.path.exists("example2.cfg"):
        config.read('example2.cfg')
        an_int = config.getint("Section1", "an_int")
        """
        try:
            an_int = config.getint("Section2", "an_int")
        except ConfigParser.NoSectionError:
            print "No such section exists..."
            pass
        
        """
    params = {"an_int" : an_int }
    print params["an_int"]       
        

config_write()
config_read()