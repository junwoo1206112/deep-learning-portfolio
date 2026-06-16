# Deployment

## Docker

`Dockerfile` and `.dockerignore` are ready at project root.

### Build and Run

```bash
docker build -t dl-portfolio-api .
docker run -p 8000:8000 dl-portfolio-api
```

### Health Check

```bash
curl http://localhost:8000/health
```

> 로컬 Docker가 없어도 GitHub Actions CI에서 자동 빌드 및 검증됩니다.

## CI/CD

GitHub Actions workflow: `.github/workflows/ci.yml`

Triggers:
- Push to `main` or `master`
- Pull request to `main` or `master`

Jobs:
1. `test`: pytest 실행 (Python 3.12)
2. `docker-build`: Docker 이미지 빌드 및 smoke test
