%question 2.5.1 and 2.5.2

X=load('C:\Users\Saif\Desktop\ms\sem 1\ML\HW5\hw5data\digit.txt');
y=load('C:\Users\Saif\Desktop\ms\sem 1\ML\HW5\hw5data\labels.txt');
sums=zeros(1,3);
pairs=zeros(3,3);
for i=1:3
    [c,tlb]=km(i*2,X,1);
    t_sum=sum_group_squares(i*2,X,tlb,c);
    sums(1,i)=t_sum;
    [p1,p2,p3]=pair_counting(y,tlb);
    pairs(i,:)=[p1 p2 p3];
end
%% 
%disp(pairs);
%disp(sums);
%% 
%question 2.5.3
sums_avg=zeros(10,10);
pairs_avg=zeros(100,3);
for l=1:10
    for i=1:10
        [c,tlb]=km(i,X,0);
        t_sum=sum_group_squares(i,X,tlb,c);
        sums_avg(l,i)=t_sum;
        [p1,p2,p3]=pair_counting(y,tlb);
        pairs_avg(l+(i-1)*10,:)=[p1 p2 p3];
    end
end
%% 
m_pairs=zeros(10,3);
for i=1:10
    m_pairs(i,:)=mean(pairs_avg((i-1)*10+1:i*10,:));
end

%% 
%question 2.5.3 and 2.5.4
%scatter(1:10,mean(sums_avg));
scatter(1:10,100*m_pairs(:,1)');
hold on;
scatter(1:10,100*m_pairs(:,2)');
hold on;
scatter(1:10,100*m_pairs(:,3)');

hold off;
%plot(1:10,m_pairs(:,1)',1:10,m_pairs(:,2)',1:10,m_pairs(:,3)');
legend('p1','p2','p3');

%% 

function [clusters,trLb] = km(k,trD,take_initial)
    if take_initial==1
        clusters=trD(1:k,:);
    else
        clusters=randi([0 255],k,size(trD,2));      %randoomly asssign cluster center
    end
    %disp(size(trD));
    trLb=zeros(size(trD,1),1);          %training labels to be generated
    prev_Lb=zeros(size(trD,1),1);
    for l=1:20                          %max iters
        %disp(l);
        prev_Lb=trLb;
        for i=1:size(trD,1)             %all data
            min_dist=100000000000000;
            for j=1:k
                dist=trD(i,:)-clusters(j,:);
                dist=dist.^2;
                d=sqrt(sum(dist,2));
                if d<min_dist           %calc dist and assign class to data
                    min_dist=d;
                    trLb(i,1)=j;
                end
            end
        end                             %assigment of labels finished
        
        for j=1:k                       %recalculating centers of clusters
            clusters(j,:)=mean(trD(find(trLb==j),:));
        end
        
        if isequal(prev_Lb,trLb)
            %disp("no change in labels,break");
            break;
        end
        
    end
end

function total_sum = sum_group_squares(k,trD,tlb,cluster)
    total_sum=0;    
    for i=1:k
        total_sum=total_sum+sum((trD(find(tlb==i),:)-cluster(i,:)).^2,'all');
    end
end

function [p1,p2,p3] = pair_counting(act_Lb,pred_Lb) 
    p1=0.0;
    p2=0.0;
    p3=0.0;
    %total=(size(act_Lb,1)*(size(act_Lb,1)-1))/2;
    c_p1=0;
    c_p2=0;
    
    for i=1:size(act_Lb,1)     
        for j=i+1:size(act_Lb,1)
            if (act_Lb(i,1)==act_Lb(j,1))&&(pred_Lb(i,1)==pred_Lb(j,1))
                c_p1=c_p1+1;
                p1=p1+1;
            elseif (act_Lb(i,1)==act_Lb(j,1))&&(pred_Lb(i,1)~=pred_Lb(j,1))
                c_p1=c_p1+1;
            elseif (act_Lb(i,1)~=act_Lb(j,1))&&(pred_Lb(i,1)~=pred_Lb(j,1))
                c_p2=c_p2+1;
                p2=p2+1;
            elseif (act_Lb(i,1)~=act_Lb(j,1))&&(pred_Lb(i,1)==pred_Lb(j,1))
                c_p2=c_p2+1;
            end
        end
    end
    p1=p1/c_p1;
    p2=p2/c_p2;
    p3=(p1+p2)/2;
    %disp(c_p1+c_p2);
    %disp(total);
end