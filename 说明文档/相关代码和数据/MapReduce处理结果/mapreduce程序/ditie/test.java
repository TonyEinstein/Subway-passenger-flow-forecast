package ditie;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class test {
	public static void main(String[] args) throws ParseException {
		String s = "2019-10-01-01.33.04.000000";
		String data = s.substring(0,10);
		Calendar calendar = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
		Date dateset = sdf.parse(data);
		calendar.setTime(dateset);
		int week = calendar.get(Calendar.DAY_OF_WEEK)-1;
		week = week==0 ? 7 : week;
		System.out.println(week);
	}
}
