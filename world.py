class world:
    def __init__(self):
        '''
        each landmark is dictionary:
        {x - float,
        y - float,
        classLabel - string,
        covaraince - 2x2 float matrix,
        index - integer}. 
        '''
        self.landmarks=[] 

    def addLandmarks(self,landmarks):
        self.landmarks.extend(landmarks)

    

    
