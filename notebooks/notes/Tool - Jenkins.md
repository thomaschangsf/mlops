// ----------------------------------------------
// Setup
// ----------------------------------------------
<-- Installation
	- Option1: Mac installer
		https://www.jenkins.io/download/lts/macos/

	- Option2: Via Docker	
		https://www.jenkins.io/doc/book/installing/docker/

<-- Progress
	- Working project:
		/Users/thomaschang/Documents/dev/git/tutorial/projects/app-generator
		
		brew services re/start jenkins-lts

		localhost:8080

	- Stuck at trying to figure how to set java variable -Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true. Alternative, point to remote git rep.


<-- Starting up
	- Jenkins app
		- common:
			* Install envinject plugin so we can set hudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT later
				manage jenkins --> available plugins --> envinject

		- via option1:
			* brew services start jenkins-lts
				set java environment variable:
					hudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true

				brew services stop jenkins-lts
				brew services edit jenkins-lts
					Environment="JAVA_OPTS=-Dhudson.model.DirectoryBrowserSupport.CSP= -Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true"

				brew services start jenkins-lts

				brew services restart jenkins-lts

			* http://localhost:8080/login?from=%2F
				password: d529075b043c4966a5d4060c2c4dd883
				password stored at /Users/thomaschang/.jenkins/secrets/initialAdminPasswor


			* Credential
				username: thomaschang
				pwd: Jenkins123$

	- Job 
		- Configure local git
			set java environment variable in terminal starting jenkins:
					java -Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true

					hudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true

			Configure tabl -> source management - > repo: 
				file:////Users/thomaschang/Documents/dev/git/tutorial/projects/app-generator

		- script
			VENV_HOME=$WORKSPACE/.venv

			# Delete previously built virtualenv
			if [ -d $VENV_HOME ]; then
			    rm -rf $VENV_HOME
			fi

			# Create virtualenv and install necessary packages
			cd $WORKSPACE
			python3 -m venv .venv
			source $VENV_HOME/bin/activate
			$VENV_HOME/bin/python3 -m pip install --upgrade pip
			$VENV_HOME/bin/pip install --quiet -r requirements.txt
			$VENV_HOME/bin/pip --quiet nosexcover

			mkdir build
			nosetests --with-xcoverage --with-xunit --cover-package=svcs --cover-erase --cover-html
			pylint -f parseable svcs/ | tee build/pylint.out

		- Click on Build now

<-- Define a new job [Python]
	- reference: http://www.alexconrad.org/2011/10/jenkins-and-python.html