import numpy as np
import tensorflow as tf

class SME(POIModel):
    
    def __init__(
        self, 
        indata,
        npoi, 
        poi_names,
        poi_defaults,
    ):
        self.npoi = npoi
        self.pois = poi_names
        self.xpoidefault = poi_defaults
        self.is_linear = False
        self.allowNegativePOI = False
        
        