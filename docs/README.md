

Training Data from:
Copyright/Legal: Used by permission of HL7 International, all rights reserved Creative Commons License

Upload a document
curl -X 'POST' \
  'http://localhost:8000/api/v1/datasets/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample.csv;type=text/csv' \
  -F 'name=test_profile' \
  -F 'description=test set'