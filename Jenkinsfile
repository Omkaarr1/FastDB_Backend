pipeline {
    agent any

    environment {
        DOCKERHUB_USERNAME = 'omkargupta702'
        IMAGE_NAME = 'nfa2'
        IMAGE_TAG = 'latest'
        FULL_IMAGE_NAME = "${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
    }

    stages {
        stage('Checkout Source') {
            steps {
                checkout scm
            }
        }

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
                    echo "Pushing Docker image to Docker Hub: ${FULL_IMAGE_NAME}"
                    docker.withRegistry('https://index.docker.io/v1/', 'docker-hub-credentials') {
                        bat "docker push ${FULL_IMAGE_NAME}"
                    }
                }
            }
        }

        stage('Deploy Service') {
            steps {
                script {
                    echo "Authenticating with Koyeb..."
                    withCredentials([string(credentialsId: 'koyeb-token', variable: 'KOYEB_API_TOKEN')]) {
                        bat "koyeb auth login --token %KOYEB_API_TOKEN%"
                    }
                    
                    echo "Redeploying service 'nfa2' with image ${FULL_IMAGE_NAME}..."
                    bat "koyeb service update nfa2 --docker ${FULL_IMAGE_NAME}"
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
