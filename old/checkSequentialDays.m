function checkSequentialDays(dates)

% dimensions
Nt = size(dates,1);
assert(size(dates,2) == 3);

% set initial year
if dates(1,2) == 1 && dates(1,3) == 1
    lastyear = dates(1,1)-1;
else
    lastyear = dates(1,1);
end

% initial month and day
ndays = [31,28,31,30,31,30,31,31,30,31,30,31];
if rem(lastyear,4) == 0; ndays(2) = 29; end
if dates(1,3) == 1
    lastmonth = dates(1,2)-1;
    lastday = ndays(lastmonth);
else
    lastmonth = dates(1,2);
    lastday = dates(1,3)-1;
end

try
    
    for t = 1:Nt
        
        % extract current date
        year = dates(t,1);
        month = dates(t,2);
        day = dates(t,3);
        
        % deal with new year
        if month == 1 && day == 1
            
            assert(12 == lastmonth);
            assert(31 == lastday);
            
        else
            
            % check year
            assert(year == lastyear);
            
            % check month and day
            ndays = [31,28,31,30,31,30,31,31,30,31,30,31];
            if rem(year,4) == 0; ndays(2) = 29; end
            if day == 1
                assert(month-1 == lastmonth);
                assert(ndays(month-1) == lastday)
            else
                assert(month == lastmonth)
                assert(day-1 == lastday);
            end
            
        end
        
        % increment
        lastyear = year;
        lastmonth = month;
        lastday = day;
        
    end
    
catch
    [t,year,month,day,lastyear,lastmonth,lastday]
    error(' ');
end


