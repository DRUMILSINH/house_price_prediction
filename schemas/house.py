from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median Income of Neighbourhood")
    HouseAge: float = Field(gt=0, description="Average age of the house")
    AveRooms: float = Field(gt=0, description="Average number of the Rooms")
    AveBedrms: float = Field(gt=0, description="Average number of the Bedrooms")
    Population: float = Field(gt=0, description="Total Population")
    AveOccup: float = Field(gt=0, description="Average number of Occupation")
    Latitude: float = Field(ge=32, le=42, description="Latitude of the house")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude of the house")


class PredictionResponse(BaseModel):
    predicted_price: str
    predicted_price_short: str
    confidence_range: str
