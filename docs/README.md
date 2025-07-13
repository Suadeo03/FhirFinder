

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

  # Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs postgres

# Connect via psql
docker exec -it fhir_postgres psql -U fhir_user -d fhir_registry

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs chroma

# Stop and remove volumes (clears all data)
docker-compose down 