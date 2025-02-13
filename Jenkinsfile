pipeline {
    agent any

    environment {
        DOCKERHUB_USERNAME = 'omkargupta702'
        IMAGE_NAME        = 'nfa2'
        IMAGE_TAG         = 'latest'
        FULL_IMAGE_NAME   = "${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
    }

    stages {
        stage('Checkout Repository') {
            steps {
                // This pulls the repository from GitHub.
                git url: 'https://github.com/Omkaarr1/FastDB_Backend', branch: 'main'
            }
        }
        
        // If the Dockerfile is located in a subdirectory, e.g., "docker",
        // uncomment the following line and wrap build/push steps in the dir block:
        //
        // dir('docker') {
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${FULL_IMAGE_NAME}"
                    // Build the Docker image using the Dockerfile in the current directory.
                    bat "docker build -t ${FULL_IMAGE_NAME} ."
                }
            }
        }
        // }
        
        stage('Push Docker Image') {
            steps {
                script {
                    echo "Logging in to Docker Hub..."
                    // Ensure DOCKER_HUB_PASSWORD is set securely in Jenkins credentials.
                    bat "docker login --username ${DOCKERHUB_USERNAME} --password %DOCKER_HUB_PASSWORD%"
                    
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
