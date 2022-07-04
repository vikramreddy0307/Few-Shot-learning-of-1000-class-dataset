//C:\Users\vsankepa\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc
//\LocalState\rootfs\home\vsankepa\projects\dcgan

// https://oldpan.me/archives/pytorch-c-libtorch-inference
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <regex>
#include <string>

#include <sys/stat.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include <experimental/algorithm>
#include <opencv2/highgui.hpp>
#include "CNNEncoder.h"
#include "RelationNetwork.h"
#include "Params.h"
#include <numeric>
#include <iterator>
#include <random>
#include <filesystem>

bool IsPathExist(const std::string &s)
{
  struct stat buffer;
  return (stat (s.c_str(), &buffer) == 0);
}



std::tuple<torch::Tensor, torch::Tensor,torch::Tensor, torch::Tensor, std::vector<int> > get_oneshot_batch(Params p)
{
    std::string path="/home/vsankepa/projects/dcgan/FSS-1000/imgs/example/support/";
    std::vector<std::string> classes_name;
    for (const auto & entry : std::filesystem::directory_iterator(path))
    
    {
        classes_name.push_back(entry.path());
    }


    const int len_classes = classes_name.size();
 
    std::vector<int> classes(len_classes);
    std::iota(classes.begin(),classes.end(),0);
    std::vector<int> choosen_classes;
    // using built-in random generator:
    std::experimental::sample(classes.begin(), classes.end(), std::back_inserter(choosen_classes),
                p.CLASS_NUM, std::mt19937{std::random_device{}()});

    
  
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 1);
    torch::Tensor support_images=torch::zeros({p.CLASS_NUM*p.SAMPLE_NUM_PER_CLASS,3,224,224},options);
    torch::Tensor support_labels=torch::zeros({p.CLASS_NUM*p.SAMPLE_NUM_PER_CLASS,p.CLASS_NUM,224,224},options);
    torch::Tensor query_images=torch::zeros({p.CLASS_NUM*p.BATCH_NUM_PER_CLASS,23,224,224},options);
    torch::Tensor query_labels=torch::zeros({p.CLASS_NUM*p.BATCH_NUM_PER_CLASS,p.CLASS_NUM,224,224},options);
    torch::Tensor zeros_tensor=torch::zeros({p.CLASS_NUM*p.BATCH_NUM_PER_CLASS,1,224,224},options);
    int class_cnt=0;
    for (auto i : choosen_classes)
    {
        std::string path="/home/vsankepa/projects/dcgan/FSS-1000/imgs/example/support/${}/label";
        path=std::regex_replace(path, std::regex("\\${}"), classes_name[i]);
        
        
        
        std::vector<std::string> imgnames;
        for (const auto & entry : std::filesystem::directory_iterator(path))
    
        {
            imgnames.push_back(entry.path());
        }
        const int len_classes = imgnames.size();
 
        std::vector<int> indexs(len_classes);
        std::iota(indexs.begin(),indexs.end(),0);
        std::vector<int> chosen_index;
        std::experimental::sample(indexs.begin(), indexs.end(), std::back_inserter(chosen_index),
                p.SAMPLE_NUM_PER_CLASS + p.BATCH_NUM_PER_CLASS, std::mt19937{std::random_device{}()});

        
        int j=0;
        for(auto k :chosen_index)
        {
            std::string img_path="/home/vsankepa/projects/dcgan/FSS-1000/imgs/example/support/"+ classes_name[i]+'/'+imgnames[k];
            cv::Mat image = cv::imread(img_path);
            if(image.empty())
            {
                std::cout << "Could not read the image: " << img_path << std::endl;
                break;
            }
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // bgr to rgb
            image = image / 255.0 ;
            cv::transposeND(image,{2,0,1},image);


             // labels
            std::string label_path="/home/vsankepa/projects/dcgan/FSS-1000/imgs/example/support/"+classes_name[i]+"/label/"+imgnames[k];
            cv::Mat label_temp = cv::imread(label_path);
            cv::Mat label_temp2[3];
            cv::split(label_temp,label_temp2);
            cv::Mat label=label_temp2[0];
            //label=label(cv::Range::all(),cv::Range::all(),cv::Range(0,1));
            if (j < p.SAMPLE_NUM_PER_CLASS)
            {
                support_images.index_put_({j},torch::from_blob(image.data,{3,image.rows,image.cols}));
                support_labels.index_put_({j,0},torch::from_blob(label.data,{label.rows,label.cols}));
            }
            else
            {
                query_images.index_put_({j-p.SAMPLE_NUM_PER_CLASS},torch::from_blob(image.data,{3,image.rows,image.cols})) ;
                query_labels.index_put_({j-p.SAMPLE_NUM_PER_CLASS,class_cnt},torch::from_blob(label.data,{label.rows,label.cols})) ;
            }
            j += 1;
        class_cnt+=1;
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 1);//.requires_grad(true);
    // torch::Tensor support_images_tensor = torch::tensor(support_images,options);
    // torch::Tensor support_labels_tensor = torch::tensor(support_labels,options);
    torch::Tensor support_images_tensor = torch::cat((support_images,support_labels), 1);

    // torch::Tensor zeros_tensor = torch::tensor(zeros,options);
    // torch::Tensor query_images_tensor = torch::tensor(query_images,options);
    torch::Tensor query_images_tensor = torch::cat((query_images,zeros_tensor), 1);
    // torch::Tensor query_labels_tensor = torch::tensor(query_labels,options);

    return std::make_tuple(support_images_tensor, support_labels, query_images_tensor, query_labels, choosen_classes);


}
}
int get_pascal_labels()
{
    int arr[][3]={{0,0,0}, { 128,0,0}, {0,128,0}, {128,128,0},
                      {0,0,128}, {128,0,128}, {0,128,128}, {128,128,128},
                      {64,0,0}, {192,0,0}, {64,128,0}, {192,128,0},
                      {64,0,128}, {192,0,128}, {64,128,128}, {192,128,128},
                     {0, 64,0}, {128, 64, 0}, {0,192,0}, {128,192,0},
                      {0,64,128}};
    return arr;

}


// at::Tensor decode_segmap(at::Tensor label_mask,int m, int n,int plot=false)
// {
//     auto label_colours = get_pascal_labels();
//     int r[m][n];
//     r = std::copy(std::begin(label_mask), std::end(label_mask), std::begin(r));
//     int g[m][n];
//     g = std::copy(std::begin(label_mask), std::end(label_mask), std::begin(g));
//     int b[m][n];
//     b = std::copy(std::begin(label_mask), std::end(label_mask), std::begin(b));
//     auto r=at::clone(label_mask);
//     auto g=at::clone(label_mask);
//     auto b=at::clone(label_mask);
//     for(int ll=0;ll<21;ll++)
//     {
//         r.index_put_({label_mask == ll},label_colours.index({ll, 0});
//         g.index_put_({label_mask == ll},label_colours.index({ll, 1});
//         b.index_put_({label_mask == ll},label_colours.index({ll, 2});
//     }
//     auto rgb=torch::zeros({m,n,3});  //,torch::dtype(torch::kInt8));
//     rgb.index_put_({"...",0},r);
//     rgb.index_put_({"...",1},g);
//     rgb.index_put_({"...",2},b);

//     return rgb;

// }
struct Params
{
    int FEATURE_DIM=64 ;
    int RELATION_DIM=8 ;
    int CLASS_NUM=1;
    int SAMPLE_NUM_PER_CLASS=5; 
    int BATCH_NUM_PER_CLASS=5;
    int EPISODE=800000;
    int TEST_EPISODE=1000;
    int LEARNING_RATE=0.001; 
    int GPU=1 ;
    int HIDDEN_UNIT=10;
    int DISPLAY_QUERY=5;
    int EXCLUDE_CLASS=6;
    std::string FEATURE_MODEL=" ";
    std::string RELATION_MODEL=" ";
   
}  ;
bool IsZero(int i)
{
  return (i!=0);
}
int main()
{
    
    Params p;
    std::cout << "VGG model:\n"<<std::endl;
    std::string vgg_path = "/home/vsankepa/projects/dcgan/FSS-100/FSS-1000/src/vgg_model.pt";
    CNNEncoder encoder(vgg_path);
    torch::jit::script::Module model= encoder->load_model();


    RelationNetwork  relation_network();
    

    // fine-tuning
    if (args.finetune)
    {    
        feature_encoder.load_state_dict(torch.load(p.FEATURE_MODEL))
        std::cout<<"load feature encoder success"<<std::endl;
        relation_network.load_state_dict(torch.load(p.RELATION_MODEL))
        std::cout<<"load relation network success"<<std::endl;
    }
    else
    {
        std::cout<<"starting from scratch"<<std::endl;
    }

    // Optimizer
    torch::optim::Adam feature_encoder_optim(
        encoder->parameters(), torch::optim::AdamOptions(p.LEARNING_RATE));
    
    torch::optim::StepLR feature_encoder_scheduler(feature_encoder_optim,p.EPISODE/10, 0.5);

    torch::optim::Adam relation_network_optim(
        relation_network->parameters(), torch::optim::AdamOptions(p.LEARNING_RATE));
    
    torch::optim::StepLR relation_network_scheduler(relation_network_optim,p.EPISODE/10, 0.5);

    std::cout<<"Training"<<std::endl;

    float last_accuracy = 0.0;

    for (int epoch = p.start_episode; epoch <= p.EPISODE; ++epoch) 
    {
        feature_encoder_scheduler.step(epoch);
        relation_network_scheduler.step(epoch);

    
        auto [samples, sample_labels, batches, batch_labels, chosen_classes]=get_oneshot_batch(p);

        //calculate features
        auto [sample_features, _] = feature_encoder->forward( torch::autograd::Variable(samples).device(torch::kCUDA, 1));
        sample_features = sample_features.view({CLASS_NUM,SAMPLE_NUM_PER_CLASS,512,7,7});
        sample_features = torch::sum(sample_features,1).squeeze(1); // 1*512*7*7
        auto [batch_features, ft_list] = feature_encoder( torch::autograd::Variable(batches).device(torch::kCUDA, 1));


        //calculate relations
        auto sample_features_ext = sample_features.unsqueeze(0).repeat({p.BATCH_NUM_PER_CLASS*p.CLASS_NUM,1,1,1,1});
        auto batch_features_ext = batch_features.unsqueeze(0).repeat({CLASS_NUM,1,1,1,1});
        batch_features_ext = at::transpose(batch_features_ext,0,1);

        auto relation_pairs = torch::cat({sample_features_ext,batch_features_ext},2).view(-1,1024,7,7);
        auto output = (relation_network->forward(relation_pairs,ft_list)).view(-1,CLASS_NUM,224,224);

        auto mse = torch::nn::MSELoss().cuda(GPU);
        auto loss = mse(output,Variable(batch_labels).device(torch::kCUDA, 1));

        // training

        feature_encoder.zero_grad();
        relation_network.zero_grad();

        loss.backward();

        torch::nn::utils::clip_grad_norm(feature_encoder.parameters(),0.5);
        torch::nn::utils::clip_grad_norm(relation_network.parameters(),0.5);

        feature_encoder_optim.step()
        relation_network_optim.step()

        if ((p.episode+1)%10 == 0)
        {
            std::cout << std::format("episode: {} loss : {}.\n", episode+1,loss.device(torch::kCPU).data<float>());
        };

        int check;
        if (IsPathExist(p.TrainResultPath)==false)
        {
            check = mkdir(p.TrainResultPath,0777);
        };
        if (IsPathExist(p.ModelSavePath)==false)
        {
            check = mkdir(p.ModelSavePath,0777);
        };

        // training result visualization
        if ((episode+1)%args.ResultSaveFreq == 0);
        {
            at::Tensor support_output=torch::zeros({224*2,224*p.SAMPLE_NUM_PER_CLASS,3},torch::dtype(torch::kInt8));
            at::Tensor query_output=torch::zeros({224*3,224*p.DISPLAY_QUERY,3},torch::dtype(torch::kInt8));
            
            std::vector <int> samplingIndex(p.BATCH_NUM_PER_CLASS);
            std::iota(samplingIndex.begin(), samplingIndex.end(), 0);
            std::sample(samplingIndex.begin(), samplingIndex.end(), std::back_inserter(chosen_query),p.DISPLAY_QUERY, std::mt19937{std::random_device{}()});

        

            for(int i=0;i<p.CLASS_NUM;i++)
            {
                for(int j=0;j<p.SAMPLE_NUM_PER_CLASS;i++)
                {
                    auto supp_img=at::transpose(samples.index({j}),1,2,0).device(torch::kCPU).to(torch::kInt32)).data< int>();
                    
                    
                    support_output.index_put_({at::indexing::Slice(0,224),at::indexing::Slice(j*224,(j+1)*224),"..."},supp_img) 
                    auto supp_label = (sample_labels.index({j})).index({0});
                    int n = sizeof(supp_label) / sizeof(supp_label[0]);
                    
                    auto size=supp_label.sizes();
                    
                    supp_label = decode_segmap(supp_label,size[0],size[0]);
                    query_output.index_put_({at::indexing::Slice(224,224*2),at::indexing::Slice(cnt*224,(cnt+1)*224)}, query_label);
                    for(int cnt=0;cnt<p.DISPLAY_QUERY;cnt++)
                    
                    {
                        
                        int x=chosen_query[cnt];
                        auto query_img=at::multiply(at::transpose(batches.index({x}),1,2,0),255).index({"...",at::indexing::Slice(0,3, 1)})
                        query_img=at::flip(query_img,{3}).index({"...",at::indexing::Slice(0,at::indexing::None, 1)});
                        query_output.index_put_({at::indexing::Slice(0,224),at::indexing::Slice(j*224,(j+1)*224),"..."},query_img)
                        auto query_label=batches.index({x})[0];
                        query_label.index_put_({query_label!=0},chosen_classes[i]);
                        query_label = decode_segmap(query_label);
                        query_output.index_put_({at::indexing::Slice(224,224*2),at::indexing::Slice(cnt*224,(cnt+1)*224),"..."},query_label);
                        auto query_pred=output[x][0];
                        query_pred=at::multiply(query_pred,255).to(torch::kInt8)
                        auto result=torch::zeros({224,224,3},torch::dtype(torch::kInt8));
                        result.index_put_({"...",0},query_pred);
                        result.index_put_({"...",1},query_pred);
                        result.index_put_({"...",2},query_pred);
                        query_output.index_put_({at::indexing::Slice(224*2,224*3),at::indexing::Slice(cnt*224,(cnt+1)*224),"..."},result);
                    }
            auto extra=at::clone(query_output);
            for(int  i=0;i<p.CLASS_NUM;i++)
            {
                int size=chosen_query.size90
                for( int cnt=0;cnt<size;cnt++)
                {
                    x=chosen_query[cnt];
                    auto extra_label =batch_labels[x][0];
                    extra_label.index_put_({extra_label!=0},255);
                    auto result1=torch::zeros({224,224,3},torch::dtype(torch::kInt8));
                    result1.index_put_({"...",0},extra_label);
                    result1.index_put_({"...",1},extra_label);
                    result1.index_put_({"...",2},extra_label);
                    extra.index_put_({at::indexing::Slice(224*2,224*3),at::indexing::Slice(cnt*224,(cnt+1)*224),"..."},result1);
            //Saving Images

            std::vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            cv::Mat query_output_mat = cv::Mat(query_output.sizes()[0], query_output.sizes()[1], CV_8UC3, query_output.to(torch::kInt32).data_ptr<int>());
            cv::Mat extra_mat = cv::Mat(extra.sizes()[0], extra.sizes()[1], CV_8UC3, extra.to(torch::kInt32).data_ptr<int>());
            cv::Mat support_output_mat = cv::Mat(support_output.sizes()[0], support_output.sizes()[1], CV_8UC3, support_output.to(torch::kInt32).data_ptr<int>());

            cv::imwrite(std::format("{}/{}_query.jpg",p.TrainResultPath,(p.episode).to_string()), query_output_mat, compression_params);
            cv::imwrite(std::format("{}/{}_show.jpg",p.TrainResultPath,(p.episode).to_string()), extra_mat, compression_params);
            cv::imwrite(std::format("{}/{}_support.jpg",p.TrainResultPath,(p.episode).to_string()), support_output_mat, compression_params);


                }
            }


        if ((episode+1) % p.ModelSaveFreq == 0)
        {
            torch::save(encoder, std::format("feature_encoder-{}-{}.pt",(p.CLASS_NUM).to_string(),(p.SAMPLE_NUM_PER_CLASS).to_string());
            torch::save(relation_network, std::format("relation_network-{}-{}.pt",(p.CLASS_NUM).to_string(),(p.SAMPLE_NUM_PER_CLASS).to_string());

        }
                    
                    
         
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 224, 224}));
    // // CNNEncoder encoder(true);
    // auto out =model.forward(inputs).toTensor();
    // std::cout<<out<<std::endl;
    
}