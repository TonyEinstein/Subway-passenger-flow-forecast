package ditie;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class map1 extends Mapper<LongWritable, Text,Text, IntWritable> {
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    if(key.toString().equals("0")) return;
    String str = value.toString();
    String[] words = str.split(",");
    String data = words[6].substring(0,10); //time
    Calendar calendar = Calendar.getInstance();
    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
    try {
      calendar.setTime(sdf.parse(data));
      int week = calendar.get(Calendar.DAY_OF_WEEK)-1;
      week = week==0 ? 7 : week; //return week
      //address
      context.write(new Text(data + " " + week + " " + words[5]),new IntWritable(1));
    } catch (ParseException e) {
      e.printStackTrace();
    }
  }
}