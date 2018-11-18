classdef km_Utils2
    methods(Static)
        function [clusters,trLb] = km(k,trD,take_initial)
            if take_initial==1
                clusters=trD(1:k,:);
            else
                clusters=randi([0 255],k,size(trD,2));      %randoomly asssign cluster center
            end
            %disp(size(trD));
            trLb=zeros(size(trD,1),1);          %training labels to be generated
            %prev_Lb=zeros(size(trD,1),1);
            for l=1:20                          %max iters
                disp(l);
                prev_Lb=trLb;
                dist=zeros(size(trD,1),k);
                for j=1:k                       %calc distances
                    dist(:,j)=sqrt(sum((trD-clusters(j,:)).^2,2));
                end
                for i=1:size(trD,1)             %calc cluster assignments
                    [m,ind]=min(dist(i,:));
                    trLb(i,1)=ind;
                end
                
                for j=1:k                       %recalculating centers of clusters
                    clusters(j,:)=mean(trD(trLb==j,:));
                end
                
                if isequal(prev_Lb,trLb)
                    %disp("no change in labels,break");
                    break;
                end
                
            end
        end
        
        function kern = l1norm(data)
            s=sum(data,1);
            kern = data ./(s+eps);
        end
        
        function [kernel] = getKernel(data)
            kernel = zeros(size(data,1),size(data,1));
            %data=km_Utils2.l1norm(data);
            for i=1:size(data,1)
                diff = data-data(i,:);                  %subtract ith row from every row
                denom = data+data(i,:);
                kernel(:,i) = sum(diff.^2 ./ (denom+eps), 2);
            end
            %kernel=km_Utils2.l1norm(kernel);                        %l1 normalization
            
        end
        %vectorized implementation referred from https://stats.stackexchange.com/questions/156069/chi-squared-kernel-and-faster-implementation
        function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
            %trainD=km_Utils2.l1norm(trainD);
            %testD=km_Utils2.l1norm(testD);                  %normalize data
            trainK=km_Utils2.getKernel(trainD);
            %testK=km_Utils2.getKernel(testD);
            
            testK=zeros(size(testD,1),size(trainD,1));
            for i=1:size(trainD,1)
                diff = testD-trainD(i,:);                  %subtract ith row from every row
                denom = testD+trainD(i,:);
                testK(:,i) = sum(diff.^2 ./ (denom+eps), 2);
            end
            %testK=km_Utils2.l1norm(testK);
            gamma_test=mean(testK,'all');
            trainK=exp(trainK*(-1/gamma));
            testK=exp(testK*(-1/gamma));
            disp(gamma_test);
        end
        
        
        function total_sum = sum_group_squares(k,trD,tlb,cluster)
            total_sum=0;
            for i=1:k
                total_sum=total_sum+sum((trD(tlb==i,:)-cluster(i,:)).^2,'all');
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
    end
end