from app import InferlessPythonModel
obj = InferlessPythonModel()
obj.initialize()

inputs = {
,"xmin":[3,2],
"ymin":[850,0],
"xmax":[1234,1098],
"ymax":[1536,504]
}
print(obj.infer(inputs))