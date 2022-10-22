package ditie;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Main {
	public static void main(String[] args) throws Exception {
		// 配置
		Configuration conf = new Configuration();
//    conf.set("fs.defaultFS","hdfs://10.16.11.230:9000");    // 使用远程HDfs作为数据源
		System.setProperty("HADOOP_USER_NAME","root");
		// 新建job
		Job job = Job.getInstance(conf);
		job.setJarByClass(Main.class);
		job.setMapperClass(map1.class);
		job.setReducerClass(reduce1.class);

		// map reduce 的 输入输出类型
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);

		// 输入输出
		// hdfs文件
//    Path input = new Path("/test.txt");
//    Path output = new Path("/out/");
		// 本地文件
		Path input = new Path("E:\\Git\\BigData\\mapreduce_test\\src\\main\\resources\\in\\acc_10_final_0.csv");
		Path output = new Path("E:\\Git\\BigData\\mapreduce_test\\src\\main\\resources\\out");
		FileInputFormat.setInputPaths(job,input);
		FileOutputFormat.setOutputPath(job,output);
		job.waitForCompletion(true);
	}
}

