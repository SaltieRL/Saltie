from bot_code.models.base_model import BaseModel

class BasePhysicsPredictor(BaseModel):
    '''
    A base class for any model that takes a subset of the gamestate
    and predicts the gamestate at a later time.
    '''
    pass
