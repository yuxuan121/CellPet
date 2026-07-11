#!/bin/sh
APP_HOME=$(cd "$(dirname "$0")" && pwd -P)
CLASSPATH="$APP_HOME/gradle/wrapper/gradle-wrapper.jar"
if [ ! -f "$CLASSPATH" ]; then
    echo "ERROR: gradle-wrapper.jar not found at $CLASSPATH"
    echo "Run: cd $APP_HOME && gradle wrapper --no-daemon"
    exit 1
fi
exec java -Xmx1024m -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"
