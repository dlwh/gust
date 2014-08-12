organization := "org.scalanlp"

name := "gust"

version := "0.1-SNAPSHOT"

scalaVersion := "2.11.1"

libraryDependencies ++= Seq(
  "junit" % "junit" % "4.5" % "test",
  "org.scalanlp" % "jcuda" % "0.5.5",
  "org.scalanlp" %% "breeze-macros" % "0.3.1" % "compile",
  "org.scalanlp" %% "breeze" % "0.9",
  "org.scalatest" %% "scalatest" % "2.1.3" % "test",
  "com.nativelibs4java" % "javacl" % "1.0-SNAPSHOT",
  "org.scalacheck" %% "scalacheck" % "1.11.3" % "test",
  "org.scala-lang.modules" %% "scala-xml" % "1.0.2" % "test",
  "org.scalanlp" % "jcublas2" % "0.5.5",
  "org.scalanlp" % "jcurand" % "0.5.5",
  "org.scalanlp" % "jcusparse2" % "0.5.5",
  "com.storm-enroute" %% "scalameter" % "0.6"
)

 fork := true

javaOptions ++= Seq("-Xmx4g", "-Xrunhprof:cpu=samples,depth=12")

resolvers ++= Seq(
  //"Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

testFrameworks += new TestFramework("org.scalameter.ScalaMeterFramework")

parallelExecution in Test := false

logBuffered := false

// NOTE: you have to disable -oDF in order for ScalaMeter to work
testOptions in Test += Tests.Argument("-oDF")
//testOptions in Test += Tests.Argument("-preJDK7")

addCompilerPlugin("org.scalamacros" %% "paradise" % "2.1.0-M1" cross CrossVersion.full)
