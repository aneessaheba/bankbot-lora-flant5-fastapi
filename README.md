## to acess tar file
```bash
docker load -i banking-classifier-api.tar
docker run -p 8000:8000 banking-classifier-api
```

##  Docker Deployment

### Build the Docker Image
```bash
docker build -t banking-classifier-api .
```

### Run the Container
```bash
docker run -p 8000:8000 banking-classifier-api
```

### Access the Application

- UI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health