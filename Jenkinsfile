pipeline {
    agent any

    environment {
        // Docker Hub environment variables
        DOCKERHUB_USERNAME = 'omkargupta702'
        IMAGE_NAME = 'nfa2'
        IMAGE_TAG = 'latest'
        // The full image name to push to Docker Hub
        FULL_IMAGE_NAME = "${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
    }

    stages {
        stage('Checkout Source') {
            steps {
                // Checkout the repository containing the Dockerfile
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${FULL_IMAGE_NAME}"
                    sh "docker build -t ${FULL_IMAGE_NAME} ."
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    echo "Pushing Docker image to Docker Hub: ${FULL_IMAGE_NAME}"
                    // Use Docker Hub credentials for login
                    docker.withRegistry('https://index.docker.io/v1/', 'docker-hub-credentials') {
                        sh "docker push ${FULL_IMAGE_NAME}"
                    }
                }
            }
        }

        stage('Deploy Service') {
            steps {
                script {
                    echo "Authenticating with Koyeb..."
                    // Login to Koyeb using the API token stored as a Jenkins credential
                    withCredentials([string(credentialsId: 'koyeb-token', variable: 'm22p4c7znj6slntrqg8rgokczqk3t4fbpf7q8h2hi6794kbth8rs3awin4ggg5r2')]) {
                        sh "koyeb auth login --token ${KOYEB_API_TOKEN}"
                    }
                    
                    echo "Redeploying service 'nfa2' with image ${FULL_IMAGE_NAME}..."
                    // Redeploy the service using the updated image
                    sh "koyeb service update nfa2 --docker ${FULL_IMAGE_NAME}"
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
