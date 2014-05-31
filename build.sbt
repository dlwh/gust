organization := "org.scalanlp"

name := "gust"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.3"

libraryDependencies ++= Seq(
  "junit" % "junit" % "4.5" % "test",
  "org.scalanlp" %% "breeze-macros" % "0.3" % "compile",
  "org.scalanlp" %% "breeze" % "0.7",
  "org.scalatest" %% "scalatest" % "2.0.M5b",
  "com.nativelibs4java" % "javacl" % "1.0-SNAPSHOT",
  "org.scalanlp" % "jcublas2" % "0.5.5",
  "org.scalanlp" % "jcurand" % "0.5.5",
  "com.github.axel22" %% "scalameter" % "0.4"
)

fork := true

javaOptions ++= Seq("-Xmx12g")

resolvers ++= Seq(
  //"Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

testFrameworks += new TestFramework("org.scalameter.ScalaMeterFramework")

parallelExecution in Test := false

logBuffered := false

// NOTE: you have to disable -oDF in order for ScalaMeter to work
//testOptions in Test += Tests.Argument("-oDF")
testOptions in Test += Tests.Argument("-preJDK7")

addCompilerPlugin("org.scala-lang.plugins" % "macro-paradise" % "2.0.0-SNAPSHOT" cross CrossVersion.full)
