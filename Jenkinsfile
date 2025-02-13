pipeline {
    agent any
    environment {
        DOCKERHUB_USERNAME = 'omkargupta702'
        IMAGE_NAME = 'nfa2'
        IMAGE_TAG = 'latest'
        FULL_IMAGE_NAME = "${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
    }
    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${FULL_IMAGE_NAME}"
                    bat "docker build -t ${FULL_IMAGE_NAME} ."
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                script {
                    // Log in to Docker Hub (ensure credentials are securely managed)
                    bat "docker login --username ${DOCKERHUB_USERNAME} --password @Omkargupta123"
                    
                    echo "Pushing Docker image: ${FULL_IMAGE_NAME}"
                    bat "docker push ${FULL_IMAGE_NAME}"
                }
            }
        }
    }
    post {
        always {
            echo "Pipeline execution completed."
        }
    }
}
