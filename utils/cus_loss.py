import warnings
import torch

warnings.filterwarnings('ignore')

def cusLoss(args, criterion, i, outputs, batch_y):
    loss = 0
    if args.rec_lambda:
        loss_rec = criterion(outputs, batch_y)
        loss += args.rec_lambda * loss_rec
        if (i + 1) % 100 == 0:
            print(f"\tloss_rec: {loss_rec.item()}")
        
    
    if args.auxi_lambda:
        # fft shape: [B, P, D]
        if args.auxi_mode == "fft":
            loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)
        
        elif args.auxi_mode == "rfft":
            if args.auxi_type == 'complex':
                loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
            elif args.auxi_type == 'complex-phase':
                loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
            elif args.auxi_type == 'complex-mag-phase':
                loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
            elif args.auxi_type == 'phase':
                loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
            elif args.auxi_type == 'mag':
                loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
            elif args.auxi_type == 'mag-phase':
                loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
            else:
                raise NotImplementedError
        
        elif args.auxi_mode == "rfft-D":
            loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)
        
        elif args.auxi_mode == "rfft-2D":
            loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)
        
        else:
            raise NotImplementedError
        
        if args.auxi_loss == "MAE":
            # MAE, 最小化element-wise error的模长
            loss_auxi = loss_auxi.abs().mean() if args.module_first else loss_auxi.mean().abs()  # check the dim of fft
        elif args.auxi_loss == "MSE":
            # MSE, 最小化element-wise error的模长
            loss_auxi = (loss_auxi.abs() ** 2).mean() if args.module_first else (loss_auxi ** 2).mean().abs()
        else:
            raise NotImplementedError
        
        loss += args.auxi_lambda * loss_auxi
        if (i + 1) % 100 == 0:
            print(f"\tloss_auxi: {loss_auxi.item()}")
        
    
        return loss
